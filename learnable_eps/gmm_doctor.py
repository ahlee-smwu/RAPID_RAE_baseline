"""Audit a gmm_clusters.pkl against the latent shards, and repair stale classes.

    # look only (steps 3-4)
    python learnable_eps/gmm_doctor.py --mode audit \
        --gmm gmm_imagenet_256/20_diag/gmm_clusters.pkl \
        --latent-path /mnt/aisha/ahlee-rae

    # audit, then refit just the stale classes and write a complete pkl (step 5)
    python learnable_eps/gmm_doctor.py --mode repair \
        --gmm gmm_imagenet_256/20_diag/gmm_clusters.pkl \
        --latent-path /mnt/aisha/ahlee-rae \
        --out gmm_imagenet_256/20_diag/gmm_clusters_repaired.pkl \
        --num-workers 8

Why this is needed
------------------
gmm_fit.py's build_class_index skips a rank whose labels_rank{R}.npy is absent:

    label_path = group_dir / f"labels_rank{rank:03d}.npy"
    if not label_path.exists():
        continue

An extraction rank that died leaves its latents_rank{R}.dat behind but never
writes labels, so the fit silently proceeded on the remaining ranks. Every
class in that group was then fitted on a fraction of its images, with no error
and no warning anywhere in the output.

The fit records ``labels[cls]`` -- one entry per sample it actually saw -- so
comparing ``len(labels[cls])`` against the number of images that class has in
the (now repaired) shards tells you exactly which classes are stale.

Repair refits only those classes, reusing gmm_fit.py's own ``process_class``
with the settings read back out of the pkl (num_components, cov_type, PCA), so
repaired classes are produced by identical code and identical hyperparameters
to the ones that were already fine. Everything else is copied through
untouched.

RAM
---
pickle.load is monolithic: this needs roughly the pkl's own size in RAM
(~63 GB for K=20, --use-pca false, since sklearn returns float64). Prefer
``--mode repair`` directly: it performs the audit internally and exits without
writing when nothing is stale, so you pay that load once rather than twice.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_fit import NUMPY_DTYPE, build_class_index, discover_groups, process_class  # noqa: E402

try:
    import threadpoolctl
except ImportError:
    threadpoolctl = None


# ----------------------------------------------------------------------
# latent side
# ----------------------------------------------------------------------

def scan_latents(latent_dir: Path):
    """class id -> list of (group_dir, rank, row_indices), plus per-class counts.

    Uses gmm_fit.py's own build_class_index so the rows gathered here are
    exactly the rows the fit would gather.
    """
    groups = discover_groups(latent_dir)
    index = defaultdict(list)
    counts = defaultdict(int)
    group_of = {}
    meta_by_group = {}

    for gdir in groups:
        with open(gdir / "meta.json") as f:
            meta_by_group[gdir] = json.load(f)
        class_ids, idx, local_n_by_rank = build_class_index(gdir)
        for cls in class_ids:
            if cls in group_of and group_of[cls] != gdir:
                raise ValueError(
                    f"class {cls} appears in both {group_of[cls].name} and {gdir.name}. "
                    "gmm_fit.py assumes each class lives in exactly one group."
                )
            group_of[cls] = gdir
            for rank, rows in idx[cls].items():
                index[cls].append((gdir, rank, rows, local_n_by_rank[rank]))
                counts[cls] += len(rows)

    return groups, dict(index), dict(counts), meta_by_group


def gather_class_rows(entries, meta_by_group) -> np.ndarray:
    """(N_c, D) float32 for one class, read straight out of the memmaps."""
    parts = []
    for gdir, rank, rows, local_n in entries:
        meta = meta_by_group[gdir]
        latent_shape = tuple(meta["latent_shape"])
        dtype = meta["dtype"]
        mm = np.memmap(gdir / f"latents_rank{rank:03d}.dat",
                       dtype=NUMPY_DTYPE[dtype], mode="r",
                       shape=(local_n, *latent_shape))
        chunk = np.asarray(mm[rows])
        if dtype == "int8":
            scale = np.load(gdir / "channel_scale.npy").astype(np.float32)
            scale = scale.reshape((1, -1) + (1,) * (len(latent_shape) - 1))
            chunk = chunk.astype(np.float32) * scale
        else:
            chunk = chunk.astype(np.float32)
        parts.append(chunk.reshape(chunk.shape[0], -1))
        del mm
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


# ----------------------------------------------------------------------
# pkl side
# ----------------------------------------------------------------------

def fit_counts(g: dict) -> dict:
    """class id -> how many samples that class's GMM was actually fitted on."""
    out = {}
    labels = g.get("labels") or {}
    resp = g.get("responsibilities") or {}
    for cls in (g.get("means") or {}):
        if cls in labels and labels[cls] is not None:
            out[cls] = int(np.asarray(labels[cls]).shape[0])
        elif cls in resp and resp[cls] is not None:
            out[cls] = int(np.asarray(resp[cls]).shape[0])
        else:
            out[cls] = -1  # unknown: force a refit rather than guess
    return out


def config_from_pkl(g: dict) -> dict:
    use_pca = g.get("pca_components") is not None
    return {
        "num_clusters": int(g["num_components"]),
        "use_pca": use_pca,
        "pca_dim": int(g["pca_dim"]) if use_pca and g.get("pca_dim") else 256,
        "cov_type": g.get("cov_type", "diag"),
    }


def audit(g: dict, counts: dict):
    fitted = fit_counts(g)
    all_classes = sorted(set(counts) | set(fitted))
    stale, missing, ok = [], [], []
    for cls in all_classes:
        now = counts.get(cls, 0)
        was = fitted.get(cls)
        if was is None:
            missing.append((cls, now))
        elif was != now:
            stale.append((cls, was, now))
        else:
            ok.append(cls)
    return ok, stale, missing, fitted


def main():
    ap = argparse.ArgumentParser(description="Audit / repair gmm_clusters.pkl against the latent shards.")
    ap.add_argument("--mode", choices=["audit", "repair"], default="audit")
    ap.add_argument("--gmm", required=True, help="gmm_clusters.pkl from gmm_fit.py")
    ap.add_argument("--latent-path", required=True, help="extract_z.py --out-dir (repaired)")
    ap.add_argument("--out", default=None, help="repair output pkl (default: <gmm>_repaired.pkl)")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="BLAS threads for the refit. Classes are refit sequentially "
                         "because only a handful are usually stale.")
    ap.add_argument("--max-classes", type=int, default=0, help="Debug: cap how many classes are refit.")
    args = ap.parse_args()

    latent_dir = Path(args.latent_path)
    print(f"[scan] {latent_dir}")
    groups, index, counts, meta_by_group = scan_latents(latent_dir)
    print(f"[scan] {len(groups)} group(s), {len(counts)} classes, "
          f"{sum(counts.values())} images")

    print(f"[load] {args.gmm}  (needs RAM comparable to the file size)")
    with open(args.gmm, "rb") as f:
        g = pickle.load(f)

    ok, stale, missing, fitted = audit(g, counts)
    cfg = config_from_pkl(g)
    print(f"[cfg]  num_clusters={cfg['num_clusters']} cov_type={cfg['cov_type']} "
          f"use_pca={cfg['use_pca']}" + (f" pca_dim={cfg['pca_dim']}" if cfg['use_pca'] else ""))

    print(f"\n  up to date : {len(ok)} classes")
    print(f"  STALE      : {len(stale)} classes (fitted on fewer images than the shards now hold)")
    print(f"  MISSING    : {len(missing)} classes (absent from the pkl entirely)")

    for cls, was, now in stale[:15]:
        pct = 100.0 * was / max(now, 1)
        print(f"     class {cls:4d}: fitted on {was} of {now} images ({pct:.1f}%)")
    if len(stale) > 15:
        print(f"     ... and {len(stale) - 15} more")
    for cls, now in missing[:15]:
        print(f"     class {cls:4d}: not in pkl, {now} images available")
    if len(missing) > 15:
        print(f"     ... and {len(missing) - 15} more")

    todo = [c for c, _, _ in stale] + [c for c, _ in missing]

    if not todo:
        print("\n" + "=" * 62)
        print("GMM IS COMPLETE — every class was fitted on all of its images.")
        print("No refit needed; go straight to convert_gmm.py.")
        return 0

    if args.mode == "audit":
        print("\n" + "=" * 62)
        print(f"NEEDS REPAIR — {len(todo)} classes.")
        print("Re-run with --mode repair to refit just those and write a complete pkl.")
        return 1

    # ---------------- repair ----------------
    if args.max_classes:
        todo = todo[: args.max_classes]
    out_path = args.out or str(Path(args.gmm).with_name(Path(args.gmm).stem + "_repaired.pkl"))

    print("\n" + "=" * 62)
    print(f"[repair] refitting {len(todo)} classes with the pkl's own settings")

    for i, cls in enumerate(todo, 1):
        if cls not in index:
            raise ValueError(
                f"class {cls} needs a refit but has no rows under {latent_dir}. "
                "Run check_latents.py: its group is probably still incomplete."
            )
        X = gather_class_rows(index[cls], meta_by_group)
        if X.shape[0] < cfg["num_clusters"]:
            raise ValueError(
                f"class {cls}: only {X.shape[0]} images for {cfg['num_clusters']} components."
            )
        if threadpoolctl is not None:
            with threadpoolctl.threadpool_limits(limits=args.num_workers):
                res = process_class(cls, X, cfg, None)
        else:
            res = process_class(cls, X, cfg, None)

        g["means"][cls] = res["means"]
        g["covs"][cls] = res["covs"]
        g["weights"][cls] = res["weights"]
        g["labels"][cls] = res["labels"]
        g["responsibilities"][cls] = res["responsibilities"]
        if res["pca_info"] is not None:
            if g.get("pca_components") is None:
                g["pca_components"] = {}
            g["pca_components"][cls] = res["pca_info"]

        del X, res
        gc.collect()
        print(f"  [{i}/{len(todo)}] class {cls} refit on {counts[cls]} images")

    # Re-audit in memory so the file we write is provably complete.
    ok2, stale2, missing2, _ = audit(g, counts)
    if stale2 or missing2:
        raise RuntimeError(
            f"after repair {len(stale2)} stale / {len(missing2)} missing classes remain; "
            "not writing a pkl that is still incomplete."
        )

    print(f"\n[write] {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(g, f, protocol=4)

    print("=" * 62)
    print(f"REPAIRED — {len(ok2)} classes, all fitted on their full image set.")
    print(f"Use this file from here on:\n  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
