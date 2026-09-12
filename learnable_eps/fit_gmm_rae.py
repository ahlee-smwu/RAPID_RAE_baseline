"""FALLBACK fitter: fit the RAPID GMM prior from scratch, encoding on the fly.

USE convert_gmm.py INSTEAD if learnable_eps/gmm_fit.py has already been run --
converting an existing gmm_clusters.pkl takes minutes and reuses the compute
already spent, whereas this script re-encodes the whole training set.

This script is the route for starting from nothing, or for refitting at a
different K without re-extracting latents.


Run in three stages::

    OUT=gmm_out_dinov2b_k10

    # 1) shared PCA basis over a class-balanced subsample
    python src/fit_gmm_rae.py --stage pca --config <cfg> --data-path <imagenet/train> \\
        --out $OUT --k 10 --pca-dim 256 --pca-per-class 8

    # 2) per-class GMM fit, shardable across GPUs
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python src/fit_gmm_rae.py --stage fit --config <cfg> \\
          --data-path <imagenet/train> --out $OUT --shard $i --num-shards 4 &
    done; wait

    # 3) merge shards, compute global_sigma_scale, write gmm_rae.pkl
    python src/fit_gmm_rae.py --stage merge --out $OUT

Why K defaults to 10
--------------------
ImageNet has ~1300 images per class. At K=10 each 256-D diagonal Gaussian is
estimated from ~130 samples (~260 with the flip augmentation); at K=20 that
drops to ~65 and the covariance estimate collapses. The K=30 used for LSUN
does not transfer -- do not raise K above 20 here.

Distribution matching
---------------------
The GMM MUST be fitted on exactly the tensor the diffusion model sees. This
repository trains on-the-fly (``z = rae.encode(images)`` in src/train.py) and
``RAE.encode`` applies the stage-1 normalisation internally, so this script
calls the same ``rae.encode`` with the same transform -- including
``RandomHorizontalFlip``, which training uses and which therefore has to be
present in the fitted distribution too (``--include-flip 1``, the default).

The ``pca`` stage prints the latent mean/std; compare it against one training
step's ``z``. They must agree to within ~1%.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, 'src'))

from stage1 import RAE  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import center_crop_arr, parse_configs  # noqa: E402


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------

def build_rae(config_path: str, device: torch.device) -> RAE:
    full_cfg = OmegaConf.load(config_path)
    rae_config = parse_configs(full_cfg)[0]
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    return rae


def build_dataset(data_path: str, image_size: int, include_flip: bool):
    """Same preprocessing as stage-2 training.

    Note ``RandomHorizontalFlip`` is a *random* transform, so a second pass
    over the dataset yields different flips. That is intended: the fitted
    distribution should cover both orientations exactly as training does.
    """
    ops = [transforms.Lambda(lambda img: center_crop_arr(img, image_size))]
    if include_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return ImageFolder(data_path, transform=transforms.Compose(ops))


def class_index_map(dataset) -> dict:
    """class id -> list of dataset indices."""
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(int(label), []).append(idx)
    return by_class


@torch.no_grad()
def encode_indices(rae, dataset, indices, device, batch_size=64, num_workers=8):
    """Encode the given dataset indices into flat latents. (N, D) float32 CPU.

    If this repository is ever switched to precomputed-latent loading, THIS is
    the function to replace -- and the replacement must read the latents from
    the same place training reads them, post-normalisation.
    """
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    out = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        z = rae.encode(images).float()
        out.append(z.reshape(z.shape[0], -1).cpu())
    if not out:
        return torch.empty(0, 0)
    return torch.cat(out, dim=0)


def shard_path(out_dir: str, shard: int) -> str:
    return os.path.join(out_dir, f"shard_{shard:04d}.pkl")


def means_memmap_path(out_dir: str) -> str:
    return os.path.join(out_dir, "means_fp16.npy")


# ----------------------------------------------------------------------
# stage 1: shared PCA basis
# ----------------------------------------------------------------------

def stage_pca(args):
    device = torch.device(args.device)
    rae = build_rae(args.config, device)
    dataset = build_dataset(args.data_path, args.image_size, bool(args.include_flip))
    by_class = class_index_map(dataset)
    num_classes = max(by_class) + 1
    print(f"[pca] dataset={len(dataset)} classes={num_classes}")

    rng = np.random.default_rng(args.seed)
    picked = []
    for c in sorted(by_class):
        idx = np.array(by_class[c])
        take = min(args.pca_per_class, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    print(f"[pca] subsampling {len(picked)} images ({args.pca_per_class}/class)")

    z = encode_indices(rae, dataset, picked, device, args.batch_size, args.num_workers)
    N, D = z.shape
    print(f"[pca] latents: N={N} D={D}")

    latent_mean = float(z.mean())
    latent_std = float(z.std())
    # RECORD THESE. They must match the training-loop z to within ~1%, and both
    # belong in the paper's table.
    print(f"[pca] latent mean={latent_mean:.6f} std={latent_std:.6f}")

    pca_mean = z.mean(dim=0)                      # (D,)
    zc = (z - pca_mean).to(device)

    # Randomized SVD on the centred matrix: N (~8000) << D (196608), so work in
    # the sample space and never form a D x D covariance.
    d = min(args.pca_dim, N - 1, D)
    q = min(N, d + args.oversample)
    g = torch.Generator(device=device).manual_seed(args.seed)
    omega = torch.randn(zc.shape[1], q, generator=g, device=device, dtype=zc.dtype)
    Y = zc @ omega                                 # (N, q)
    Q, _ = torch.linalg.qr(Y)                      # (N, q)
    for _ in range(args.power_iters):
        Q, _ = torch.linalg.qr(zc.t() @ Q)         # (D, q)
        Q, _ = torch.linalg.qr(zc @ Q)             # (N, q)
    B = Q.t() @ zc                                 # (q, D)
    _, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Vh[:d].contiguous()                        # (d, D) orthonormal rows
    sing = S[:d]

    total_var = float((zc ** 2).sum())
    evr = (sing ** 2).sum().item() / max(total_var, 1e-12)
    print(f"[pca] pca_dim={d} explained_variance_ratio={evr:.6f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(
        {
            "pca_U": U.cpu(),
            "pca_mean": pca_mean.cpu(),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "explained_variance_ratio": evr,
            "num_classes": num_classes,
            "D": D,
            "pca_dim": d,
            "K": args.k,
            "image_size": args.image_size,
            "include_flip": bool(args.include_flip),
        },
        os.path.join(args.out, "pca.pt"),
    )
    print(f"[pca] wrote {os.path.join(args.out, 'pca.pt')}")


# ----------------------------------------------------------------------
# stage 2: per-class GMM fit (shardable)
# ----------------------------------------------------------------------

def stage_fit(args):
    from sklearn.mixture import GaussianMixture

    device = torch.device(args.device)
    pca = torch.load(os.path.join(args.out, "pca.pt"), map_location="cpu")
    U = pca["pca_U"].to(device)                    # (d, D)
    pca_mean = pca["pca_mean"].to(device)          # (D,)
    D = int(pca["D"])
    d = int(pca["pca_dim"])
    K = int(args.k or pca["K"])

    rae = build_rae(args.config, device)
    dataset = build_dataset(args.data_path, args.image_size, bool(pca["include_flip"]))
    by_class = class_index_map(dataset)
    classes = sorted(by_class)
    mine = [c for i, c in enumerate(classes) if i % args.num_shards == args.shard]
    if args.max_classes:
        mine = mine[: args.max_classes]
    print(f"[fit shard {args.shard}/{args.num_shards}] {len(mine)} classes, K={K}")

    result = {}
    for n, c in enumerate(mine):
        idx = by_class[c]
        if args.max_per_class:
            idx = idx[: args.max_per_class]
        z = encode_indices(rae, dataset, idx, device, args.batch_size, args.num_workers)
        Nc = z.shape[0]
        k_eff = max(1, min(K, Nc // args.min_samples_per_cluster))
        if k_eff < K:
            print(f"[fit] class {c}: only {Nc} samples -> K reduced {K}->{k_eff}")

        zg = z.to(device)
        z_pca = (zg - pca_mean) @ U.t()             # (Nc, d)

        gm = GaussianMixture(
            n_components=k_eff,
            covariance_type="diag",
            reg_covar=args.reg_covar,
            max_iter=args.max_iter,
            n_init=1,
            random_state=args.seed + c,
        )
        z_pca_np = z_pca.cpu().numpy().astype(np.float64)
        gm.fit(z_pca_np)

        resp = torch.from_numpy(gm.predict_proba(z_pca_np)).to(device=device, dtype=torch.float32)

        # Cluster means are taken as RESPONSIBILITY-WEIGHTED FULL-SPACE means,
        # not as PCA reconstructions. The discarded PCA complement still
        # carries class-identifying signal, and the prior's whole job is to
        # deliver that signal.
        wsum = resp.sum(dim=0).clamp(min=1e-8)      # (k_eff,)
        means_full = (resp.t() @ zg) / wsum.unsqueeze(1)   # (k_eff, D)

        means_pca = torch.from_numpy(gm.means_).to(torch.float32)
        covs_pca = torch.from_numpy(gm.covariances_).to(torch.float32)
        weights = torch.from_numpy(gm.weights_).to(torch.float32)

        # Pad up to K so every class has a uniform layout. Padded components
        # get weight 0 so they can never be selected.
        if k_eff < K:
            pad = K - k_eff
            means_pca = torch.cat([means_pca, torch.zeros(pad, d)], dim=0)
            covs_pca = torch.cat([covs_pca, torch.ones(pad, d)], dim=0)
            weights = torch.cat([weights, torch.zeros(pad)], dim=0)
            means_full = torch.cat([means_full, torch.zeros(pad, D, device=device)], dim=0)

        result[c] = {
            "means_pca": means_pca,
            "covs_pca": covs_pca,
            "weights": weights,
            "means_full": means_full.to(torch.float16).cpu().numpy(),
            "n_samples": Nc,
            "k_eff": k_eff,
        }
        if (n + 1) % 10 == 0 or n == len(mine) - 1:
            print(f"[fit shard {args.shard}] {n + 1}/{len(mine)} classes done")

    with open(shard_path(args.out, args.shard), "wb") as f:
        pickle.dump(result, f, protocol=4)
    print(f"[fit shard {args.shard}] wrote {shard_path(args.out, args.shard)}")


# ----------------------------------------------------------------------
# stage 3: merge
# ----------------------------------------------------------------------

def stage_merge(args):
    pca = torch.load(os.path.join(args.out, "pca.pt"), map_location="cpu")
    U = pca["pca_U"]                               # (d, D)
    D = int(pca["D"])
    d = int(pca["pca_dim"])
    K = int(args.k or pca["K"])
    num_classes = int(pca["num_classes"])

    shards = sorted(Path(args.out).glob("shard_*.pkl"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pkl under {args.out}")
    merged = {}
    for sp in shards:
        with open(sp, "rb") as f:
            merged.update(pickle.load(f))
    print(f"[merge] {len(shards)} shards -> {len(merged)} classes (expected {num_classes})")
    missing = [c for c in range(num_classes) if c not in merged]
    if missing:
        raise ValueError(f"Missing classes in shards: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    means_pca = torch.stack([merged[c]["means_pca"] for c in range(num_classes)])   # (C, K, d)
    covs_pca = torch.stack([merged[c]["covs_pca"] for c in range(num_classes)])     # (C, K, d)
    weights = torch.stack([merged[c]["weights"] for c in range(num_classes)])       # (C, K)

    # Full-space means -> fp16 memmap on disk. Keep this file on LOCAL SSD:
    # in mmap mode training reads B x D x 2B per batch (25 MB at B=64), which
    # becomes the bottleneck over NFS.
    mm_path = means_memmap_path(args.out)
    mm = np.lib.format.open_memmap(
        mm_path, mode="w+", dtype=np.float16, shape=(num_classes, K, D)
    )
    for c in range(num_classes):
        mm[c] = merged[c]["means_full"]
    mm.flush()
    del mm
    size_gb = os.path.getsize(mm_path) / 1e9
    print(f"[merge] wrote {mm_path} ({size_gb:.2f} GB)")

    # global_sigma_scale: one scalar, fixed for the whole run, that puts the
    # mixture-weighted mean data-space variance at 1 so the GMM draw is on the
    # same scale as N(0, I). Same formula as RAPID's precompute_sigma_scale,
    # with the single shared basis.
    U2 = U * U                                     # (d, D)
    v = torch.clamp(covs_pca, min=1e-8)            # (C, K, d)
    # mean over D of (v @ U2) == v @ (U2.mean(dim=1)) -- avoids materialising
    # a (C*K, D) tensor, and is exact, not an approximation.
    u2_mean = U2.mean(dim=1)                       # (d,)
    mean_var_per_cluster = v @ u2_mean             # (C, K)
    weighted = (weights * mean_var_per_cluster).sum()
    wsum = weights.sum()
    global_mean_var = (weighted / (wsum + 1e-8)).item()
    global_sigma_scale = 1.0 / math.sqrt(global_mean_var + 1e-8)
    # RECORD THIS -- it goes in the paper table.
    print(f"[merge] global_mean_var={global_mean_var:.6e} global_sigma_scale={global_sigma_scale:.6f}")

    n_samples = [merged[c]["n_samples"] for c in range(num_classes)]
    k_eff = [merged[c]["k_eff"] for c in range(num_classes)]
    print(f"[merge] samples/class min={min(n_samples)} max={max(n_samples)} "
          f"mean={sum(n_samples)/len(n_samples):.1f}")
    print(f"[merge] k_eff min={min(k_eff)} max={max(k_eff)}")

    c_lat, h, w = 768, 16, 16
    if D != c_lat * h * w:
        # Derive the layout from D when the config is not the 768x16x16 default.
        side = int(round(math.sqrt(D / c_lat)))
        h = w = side
        if c_lat * h * w != D:
            raise ValueError(f"Cannot infer latent layout for D={D}.")

    ckpt = {
        "pca_U": U.numpy().astype(np.float32),
        "pca_mean": pca["pca_mean"].numpy().astype(np.float32),
        "means_pca": means_pca.numpy().astype(np.float32),
        "covs_pca": covs_pca.numpy().astype(np.float32),
        "weights": weights.numpy().astype(np.float32),
        "means_file": os.path.basename(mm_path),
        "means_shape": (num_classes, K, D),
        "latent_shape": (c_lat, h, w),
        "num_classes": num_classes,
        "K": K,
        "pca_dim": d,
        "global_sigma_scale": global_sigma_scale,
        "latent_mean": pca["latent_mean"],
        "latent_std": pca["latent_std"],
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "include_flip": pca["include_flip"],
    }
    out_pkl = os.path.join(args.out, "gmm_rae.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(ckpt, f, protocol=4)
    print(f"[merge] wrote {out_pkl}")


# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit the RAPID GMM prior on RAE latents.")
    p.add_argument("--stage", required=True, choices=["pca", "fit", "merge"])
    p.add_argument("--config", type=str, default=None, help="Stage-2 training YAML (needs stage_1).")
    p.add_argument("--data-path", type=str, default=None, help="ImageFolder root (ImageNet train).")
    p.add_argument("--out", type=str, required=True, help="Output directory.")
    p.add_argument("--k", type=int, default=10,
                   help="GMM components per class. Do not exceed 20 on ImageNet.")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--pca-per-class", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    p.add_argument("--include-flip", type=int, default=1,
                   help="Match the training RandomHorizontalFlip. Turning this off biases assignments.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oversample", type=int, default=16)
    p.add_argument("--power-iters", type=int, default=2)
    p.add_argument("--reg-covar", type=float, default=1e-6)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--min-samples-per-cluster", type=int, default=32)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-classes", type=int, default=0,
                   help="Debug: cap classes processed by this shard (0 = all).")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="Debug: cap images per class (0 = all).")
    args = p.parse_args()
    if args.k > 20:
        raise ValueError(
            f"--k {args.k} is too large for ImageNet (~1300 images/class); "
            "the diagonal covariance estimate collapses. Keep K <= 20."
        )
    if args.stage in ("pca", "fit"):
        if not args.config or not args.data_path:
            raise ValueError(f"--config and --data-path are required for stage {args.stage}.")
    return args


if __name__ == "__main__":
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "pca":
        stage_pca(a)
    elif a.stage == "fit":
        stage_fit(a)
    else:
        stage_merge(a)
