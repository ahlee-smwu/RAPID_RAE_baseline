"""Convert gmm_fit.py's ``gmm_clusters.pkl`` into a RAPID prior checkpoint.

Use this when the GMM has ALREADY been fitted with ``learnable_eps/gmm_fit.py``
-- there is no need to refit. ``fit_gmm_rae.py`` is only for starting from
scratch.

    python learnable_eps/convert_gmm.py \
        --gmm gmm_imagenet_256/20_diag/gmm_clusters.pkl \
        --out gmm_prior_k20 \
        --latent-shape 768 16 16

What gmm_fit.py produces (the "9-key" dict)
-------------------------------------------
    means              {cls: (K, D)}   ALWAYS data space
                                       (with --use-pca it is pca.inverse_transform'd back to D)
    covs               {cls: (K, *)}   diagonal variances in the FIT space:
                                         --use-pca false -> (K, D), data space
                                         --use-pca true  -> (K, pca_dim), PCA space
    weights            {cls: (K,)}
    labels             {cls: (N_c,)}   per-sample hard assignment  (not needed here)
    responsibilities   {cls: (N_c, K)} (not needed here)
    pca_components     {cls: {components, mean}} or None
    num_components, pca_dim, cov_type

What the prior needs, and why it differs
----------------------------------------
The POSTERIOR must score all K clusters of a class at once. In full space that
is B*K*D*2B of reads per training step (500 MB at B=64, K=20) -- unusable. So
this script builds a SHARED PCA basis and projects the per-class GMM into it,
exactly mirroring how RAPID's ``get_cluster_gmm`` scored in PCA space.

The DRAW touches only the one selected cluster -- two (B, D) gathers, ~50 MB
per step -- so means AND variances are kept in FULL space, fp16, and the draw
stays exact. Nothing about the sampled x0_gmm is approximated; only the
argmax/multinomial choice of k goes through the projection.

The shared basis is fitted on the CLUSTER MEANS (C*K vectors, e.g. 20,000),
not on raw data: it only has to separate cluster centres well enough to pick
the right k.

Covariance projection: for z = U(x - m), Cov(z) = U diag(v) U^T, whose
diagonal is (U**2) @ v. Off-diagonal terms are dropped -- standard for a
diagonal-covariance model, and it only affects cluster choice, not the draw.

RAM
---
``pickle.load`` is monolithic, so this needs roughly as much RAM as the pkl is
big. With --use-pca false, K=20, D=196608 and sklearn's float64 output that is
means 31 GB + covs 31 GB ~= 63 GB. Run it on a big-RAM node. The script drops
``labels`` and ``responsibilities`` immediately after load, and streams each
class straight out to the memmaps, so nothing is duplicated beyond the pkl.

Output (--out DIR):
    gmm_rae.pkl      small: shared basis + projected GMM + scalars
    means_fp16.npy   (C, K, D) fp16   exact full-space cluster means
    vars_fp16.npy    (C, K, D) fp16   exact full-space diagonal variances
Keep both .npy files on LOCAL SSD.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle

import numpy as np
import torch


def _as_U_dD(components: np.ndarray, D: int) -> np.ndarray:
    """Return PCA components as (d, D), whichever way round they were stored."""
    a, b = components.shape
    if b == D:
        return components
    if a == D:
        return components.T
    raise ValueError(f"PCA components {components.shape} match neither (d,{D}) nor ({D},d).")


def main():
    ap = argparse.ArgumentParser(
        description="Convert gmm_fit.py output into a RAPID prior checkpoint."
    )
    ap.add_argument("--gmm", required=True, help="Path to gmm_clusters.pkl from gmm_fit.py.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--latent-shape", type=int, nargs=3, default=[768, 16, 16],
                    metavar=("C", "H", "W"), help="Latent layout; product must equal D.")
    ap.add_argument("--pca-dim", type=int, default=256,
                    help="Shared basis dimension used for cluster assignment only.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--var-floor", type=float, default=1e-8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    print(f"[load] {args.gmm}  (this needs RAM comparable to the file size)")
    with open(args.gmm, "rb") as f:
        g = pickle.load(f)

    # Not needed downstream, and they are the second-largest entries.
    g.pop("labels", None)
    g.pop("responsibilities", None)

    means = g["means"]
    covs = g["covs"]
    weights = g["weights"]
    pca_components = g.get("pca_components", None)
    cov_type = g.get("cov_type", "diag")
    if cov_type != "diag":
        raise NotImplementedError(
            f"cov_type={cov_type!r} is not supported; the prior assumes diagonal "
            "covariances (refit gmm_fit.py with --cov-type diag)."
        )

    classes = sorted(means.keys())
    num_classes = len(classes)
    if classes != list(range(num_classes)):
        raise ValueError(
            f"Class ids must be dense 0..{num_classes-1}; got min={classes[0]} "
            f"max={classes[-1]} n={num_classes}."
        )

    K = int(np.asarray(weights[classes[0]]).shape[0])
    D = int(np.asarray(means[classes[0]]).shape[1])
    c_lat, h, w = args.latent_shape
    if c_lat * h * w != D:
        raise ValueError(f"--latent-shape {args.latent_shape} has product {c_lat*h*w} != D={D}.")

    used_pca = pca_components is not None
    print(f"[info] classes={num_classes} K={K} D={D} cov_type={cov_type} use_pca={used_pca}")
    if K > 20:
        print(f"[warn] K={K}: with ~1300 images/class each component is estimated from "
              f"~{1300//K} samples. The diagonal covariance is poorly conditioned; "
              f"consider refitting at K=10.")

    # ---------------- pass 1: full-space means and variances -------------
    means_mm = np.lib.format.open_memmap(
        os.path.join(args.out, "means_fp16.npy"), mode="w+",
        dtype=np.float16, shape=(num_classes, K, D))
    vars_mm = np.lib.format.open_memmap(
        os.path.join(args.out, "vars_fp16.npy"), mode="w+",
        dtype=np.float16, shape=(num_classes, K, D))

    weights_out = np.zeros((num_classes, K), dtype=np.float32)

    print("[pass 1] writing full-space means + variances to fp16 memmaps")
    for c in classes:
        m = np.asarray(means[c], dtype=np.float32)              # (K, D)
        if m.shape != (K, D):
            raise ValueError(f"class {c}: means {m.shape} != {(K, D)}")

        v = np.asarray(covs[c], dtype=np.float32)
        if v.shape == (K, D):
            # --use-pca false: already data-space diagonal variances.
            v_full = v
        elif used_pca and c in pca_components:
            # --use-pca true: restore data-space diagonals as v_k @ U**2.
            # This is exactly the table RAPID's precompute_sigma_scale built.
            U_c = _as_U_dD(np.asarray(pca_components[c]["components"], dtype=np.float32), D)
            v_full = np.clip(v, args.var_floor, None) @ (U_c * U_c)
        else:
            raise ValueError(
                f"class {c}: covs shape {v.shape} is neither (K,D)={(K,D)} nor "
                f"accompanied by a PCA basis to restore it from."
            )

        means_mm[c] = m.astype(np.float16)
        vars_mm[c] = np.clip(v_full, args.var_floor, None).astype(np.float16)

        wt = np.asarray(weights[c], dtype=np.float64)
        weights_out[c] = (wt / max(wt.sum(), 1e-12)).astype(np.float32)

        if (c + 1) % 100 == 0 or c == classes[-1]:
            print(f"  {c + 1}/{num_classes} classes")

    means_mm.flush()
    vars_mm.flush()

    # global_sigma_scale: one scalar fixed for the whole run, putting the
    # mixture-weighted mean data-space variance at 1 so the GMM draw sits on
    # the same scale as N(0, I). Same definition as RAPID's
    # precompute_sigma_scale, evaluated on the exact full-space table.
    wsum = 0.0
    weighted = 0.0
    for c in classes:
        per_cluster_mean_var = np.asarray(vars_mm[c], dtype=np.float32).mean(axis=1)  # (K,)
        weighted += float((weights_out[c] * per_cluster_mean_var).sum())
        wsum += float(weights_out[c].sum())
    global_mean_var = weighted / max(wsum, 1e-12)
    global_sigma_scale = 1.0 / math.sqrt(global_mean_var + 1e-8)
    # RECORD THIS -- it goes in the paper table.
    print(f"[stats] global_mean_var={global_mean_var:.6e} "
          f"global_sigma_scale={global_sigma_scale:.6f}")

    # ---------------- pass 2: shared PCA basis over cluster means --------
    print(f"[pass 2] fitting shared PCA basis (d={args.pca_dim}) on {num_classes*K} cluster means")
    M = torch.from_numpy(
        np.asarray(means_mm, dtype=np.float32).reshape(num_classes * K, D)
    ).to(device)
    pca_mean = M.mean(dim=0)
    Mc = M - pca_mean

    d = min(args.pca_dim, Mc.shape[0] - 1, D)
    # Randomized range finder: N = C*K (~20k) << D, so work in sample space.
    gen = torch.Generator(device=device).manual_seed(0)
    omega = torch.randn(D, min(Mc.shape[0], d + 16), generator=gen, device=device)
    Q, _ = torch.linalg.qr(Mc @ omega)
    for _ in range(2):
        Q, _ = torch.linalg.qr(Mc.t() @ Q)
        Q, _ = torch.linalg.qr(Mc @ Q)
    B = Q.t() @ Mc
    _, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Vh[:d].contiguous()                       # (d, D), orthonormal rows

    evr = float((S[:d] ** 2).sum() / max(float((Mc ** 2).sum()), 1e-12))
    print(f"[stats] shared basis d={d} explained_variance_ratio(cluster means)={evr:.6f}")

    # ---------------- pass 3: project the GMM into the shared basis ------
    print("[pass 3] projecting per-class GMM into the shared basis")
    means_pca = np.zeros((num_classes, K, d), dtype=np.float32)
    covs_pca = np.zeros((num_classes, K, d), dtype=np.float32)
    U2 = (U * U)                                  # (d, D)

    for c in classes:
        m = torch.from_numpy(np.asarray(means_mm[c], dtype=np.float32)).to(device)   # (K, D)
        v = torch.from_numpy(np.asarray(vars_mm[c], dtype=np.float32)).to(device)    # (K, D)
        means_pca[c] = ((m - pca_mean) @ U.t()).cpu().numpy()
        # diag(U diag(v) U^T) = (U**2) @ v
        covs_pca[c] = (v @ U2.t()).clamp(min=args.var_floor).cpu().numpy()
        if (c + 1) % 200 == 0 or c == classes[-1]:
            print(f"  {c + 1}/{num_classes} classes")

    lat_mean = float(np.asarray(means_mm[0], dtype=np.float32).mean())
    lat_std = float(np.sqrt(global_mean_var))

    ckpt = {
        "pca_U": U.cpu().numpy().astype(np.float32),
        "pca_mean": pca_mean.cpu().numpy().astype(np.float32),
        "means_pca": means_pca,
        "covs_pca": covs_pca,
        "weights": weights_out,
        "means_file": "means_fp16.npy",
        "vars_file": "vars_fp16.npy",
        "means_shape": (num_classes, K, D),
        "latent_shape": (c_lat, h, w),
        "num_classes": num_classes,
        "K": K,
        "pca_dim": d,
        "global_sigma_scale": global_sigma_scale,
        "latent_mean": lat_mean,
        "latent_std": lat_std,
        "explained_variance_ratio": evr,
        "source": os.path.abspath(args.gmm),
        "source_used_pca": used_pca,
    }
    out_pkl = os.path.join(args.out, "gmm_rae.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(ckpt, f, protocol=4)

    mb = lambda p: os.path.getsize(os.path.join(args.out, p)) / 1e9
    print(f"\n[done] {out_pkl}")
    print(f"       means_fp16.npy {mb('means_fp16.npy'):.2f} GB")
    print(f"       vars_fp16.npy  {mb('vars_fp16.npy'):.2f} GB")
    print(f"       -> set prior.ckpt_path: {out_pkl}")
    print( "       keep the two .npy files next to the pkl, on LOCAL SSD")


if __name__ == "__main__":
    main()
