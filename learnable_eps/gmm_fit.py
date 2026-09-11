"""
gmm_fit.py

Fit per-class Gaussian Mixture Models (default: k=20 components/class) on RAE
encoder latents that were extracted and cached by extract_latents.py.

Reuses the exact same GMM-fitting math (process_class()) as the previous
ImgLatentDataset-based gmm fit script. What changed across iterations of
this script is entirely about HOW the input features for each class are
gathered -- see the two design notes below.

Latent source format (extract_latents.py's memmap shard layout):
    <latent_dir>/group{NNN}/meta.json
    <latent_dir>/group{NNN}/latents_rank{R:03d}.dat   (memmap, shape (N_r, *latent_shape))
    <latent_dir>/group{NNN}/labels_rank{R:03d}.npy    (int64 class id per sample)
    <latent_dir>/group{NNN}/global_index_rank{R:03d}.npy  (ImageFolder index; unused here)
    <latent_dir>/group{NNN}/channel_scale.npy         (only if meta['dtype'] == 'int8')
--latent-dir is the PARENT folder holding all group000/, group001/, ... (or,
if extraction wasn't grouped, a folder with a single top-level meta.json).

---------------------------------------------------------------------------
NO DISK CACHE: read each class directly out of the original shards
---------------------------------------------------------------------------
Earlier versions of this script re-materialized every sample as fp32 into
per-class .bin files before fitting (Phase 1 "caching"), which needed as
much EXTRA disk as the entire extracted dataset (~968 GB here) -- first for
a whole group at a time (~242 GB), which still didn't fit comfortably in
available disk.

This version doesn't cache anything to disk at all:
  1. build_class_index(group_dir) reads ONLY the small labels_rank*.npy
     files and builds, for each class id, a mapping {rank: row_indices}
     recording exactly which rows of which rank shard belong to that class.
     This never touches the large latents_rank*.dat memmaps and is nearly
     free (a few MB of int64 labels per group).
  2. For each class, one at a time: open each relevant rank's memmap
     read-only and fancy-index it at that class's row_indices --
     `mm[row_indices]` -- which reads exactly that class's rows (and nothing
     else) off disk into a fresh in-memory array. Dequantize (if int8),
     flatten, fit the GMM, pickle the small per-class result, then drop the
     array and move to the next class.

Total bytes read off disk over a full run are the same as before (every
sample read exactly once) -- what's eliminated is the second full fp32 copy
that used to sit on disk while fitting. Peak transient disk usage now is
just the small per-class result pickles (see PEAK DISK/RAM note below) that
accumulate in temp_gmm_pkls/ until the final merge, at which point they are
deleted, leaving only the final gmm_clusters.pkl.

Peak transient RAM ~= --num-workers x (avg samples/class in this dataset x
bytes/sample), since that many classes' raw arrays can be live at once, one
per worker process. inspect_groups() prints this estimate.
---------------------------------------------------------------------------
GROUP-BY-GROUP + CLASS-BY-CLASS + no-overlap safety check
---------------------------------------------------------------------------
inspect_groups() confirmed each group{NNN}/ holds a disjoint class range
(no class spans more than one group). do_gmm_clustering() still walks
groups one at a time (so a group's class index -- and any open memmaps --
only exist while that group is being processed), and within each group
processes classes one at a time (optionally in parallel across
--num-workers processes). It re-checks the no-overlap assumption every run
and raises if it ever stops holding, since a class split across groups
would otherwise get fit on an incomplete slice of its data.
---------------------------------------------------------------------------
PARALLELISM / BLAS oversubscription
---------------------------------------------------------------------------
--num-workers processes fit different classes concurrently. Since
scikit-learn's GMM path calls into BLAS internally, each worker pins its own
BLAS thread count to 1 via threadpoolctl (no-op if not installed) so
N-processes x M-BLAS-threads doesn't oversubscribe the machine.

process_class() itself -- the actual GMM fitting math -- is untouched from
the original script.
---------------------------------------------------------------------------

Usage:
    python gmm_fit_from_rae_latents.py --inspect-only          # check first
    python gmm_fit_from_rae_latents.py --num-clusters 20 --num-workers 16
"""

import os
import sys
import gc
import json
import shutil
import pickle
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

try:
    import threadpoolctl
except ImportError:
    threadpoolctl = None

# Matches extract_latents.py's NUMPY_DTYPE table exactly (bf16 has no native
# numpy dtype, so it's stored on disk as fp32 there too).
NUMPY_DTYPE = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "int8": np.int8}


def process_class(cls, X, g_cfg, km_data):
    """
    Unchanged from the previous script: fit one class's GMM and extract the
    same set of outputs (labels, responsibilities, weights, means, covs).
    """
    num_components = g_cfg['num_clusters']
    use_pca = g_cfg.get('use_pca', False)
    pca_dim = g_cfg.get('pca_dim', 256)
    cov_type = g_cfg.get('cov_type', 'diag')

    res = {}

    if use_pca:
        pca = PCA(n_components=pca_dim)
        X_input = pca.fit_transform(X)
        res["pca_info"] = {"components": pca.components_, "mean": pca.mean_}
    else:
        X_input = X
        res["pca_info"] = None

    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type=cov_type,
        random_state=0,
        reg_covar=1e-6,
        max_iter=200,
        init_params='kmeans' if km_data is None else 'random',
        verbose=1
    )

    if km_data is not None and cls in km_data['centers']:
        init_means = km_data['centers'][cls]
        if use_pca:
            init_means = pca.transform(init_means)
        gmm.means_init = init_means

    gmm.fit(X_input)

    res["labels"] = gmm.predict(X_input)
    res["responsibilities"] = gmm.predict_proba(X_input)
    res["weights"] = gmm.weights_

    if use_pca:
        res["means"] = pca.inverse_transform(gmm.means_)
        res["covs"] = gmm.covariances_
    else:
        res["means"] = gmm.means_
        res["covs"] = gmm.covariances_

    return res


def discover_groups(latent_dir):
    """
    Returns a list of directories, each holding one meta.json plus its own
    latents_rank*/labels_rank*/(channel_scale.npy) shards. Handles both the
    grouped layout (group000/, group001/, ...) and the ungrouped layout
    (single top-level meta.json) transparently.
    """
    latent_dir = Path(latent_dir)
    group_metas = sorted(latent_dir.glob("group*/meta.json"))
    if group_metas:
        return [m.parent for m in group_metas]
    if (latent_dir / "meta.json").exists():
        return [latent_dir]
    raise FileNotFoundError(
        f"No meta.json found directly under {latent_dir} or under {latent_dir}/group*/ -- "
        f"is this really an extract_latents.py --out-dir?"
    )


def inspect_groups(latent_dir):
    """
    Cheap pre-flight report (labels only, never touches the big latent
    memmaps): per-group sample/class counts, observed vs. expected class
    range, cross-group class overlap, and a peak-RAM estimate for the
    class-by-class fitting approach (no disk cache is built anymore, so
    there's no disk-usage estimate to give here beyond "negligible").
    """
    groups = discover_groups(latent_dir)
    print(f"\nInspecting {len(groups)} group(s) under {latent_dir}\n")
    header = f"{'group':16s} {'samples':>10s} {'classes':>8s} {'observed range':>16s} {'expected range (meta)':>22s} {'avg class MB':>12s}"
    print(header)
    print("-" * len(header))

    total_samples = 0
    per_group_class_sets = []
    max_avg_class_bytes = 0

    for group_dir in groups:
        with open(group_dir / "meta.json") as f:
            meta = json.load(f)
        world_size = meta["world_size"]
        latent_shape = tuple(meta["latent_shape"])
        bytes_per_sample_fp32 = int(np.prod(latent_shape)) * 4  # always fp32 once loaded into RAM

        class_ids = set()
        n_samples = 0
        for rank in range(world_size):
            label_path = group_dir / f"labels_rank{rank:03d}.npy"
            if not label_path.exists():
                continue
            labels = np.load(label_path, mmap_mode="r")
            n_samples += labels.shape[0]
            class_ids.update(np.unique(labels).tolist())

        avg_class_bytes = (n_samples / len(class_ids) * bytes_per_sample_fp32) if class_ids else 0
        max_avg_class_bytes = max(max_avg_class_bytes, avg_class_bytes)

        expected = meta.get("class_group")
        expected_str = f"[{expected['class_range'][0]}, {expected['class_range'][1]})" if expected else "n/a (ungrouped)"
        observed_str = f"[{min(class_ids)}, {max(class_ids)}]" if class_ids else "-"

        print(f"{group_dir.name:16s} {n_samples:>10d} {len(class_ids):>8d} {observed_str:>16s} "
              f"{expected_str:>22s} {avg_class_bytes / 1e6:>11.1f}M")

        total_samples += n_samples
        per_group_class_sets.append(class_ids)

    union_classes = set().union(*per_group_class_sets) if per_group_class_sets else set()
    summed_per_group = sum(len(s) for s in per_group_class_sets)
    overlap = summed_per_group - len(union_classes)

    print("-" * len(header))
    print(f"{'TOTAL':16s} {total_samples:>10d} {len(union_classes):>8d}")
    print(f"\nThis version reads each class directly from the original shards and fits it in "
          f"memory -- no fp32 disk cache is built, so disk usage stays negligible (just the small "
          f"per-class result pickles, deleted after the final merge).")
    print(f"Peak transient RAM with --num-workers=W is roughly W x {max_avg_class_bytes / 1e6:.0f} MB "
          f"(largest group's average class size) -- e.g. W=8 -> ~{8 * max_avg_class_bytes / 1e9:.1f} GB.")

    if overlap > 0:
        print(f"⚠️  {overlap} class id(s) appear in more than one group -- this script processes one "
              f"group at a time and would fit each such class on an incomplete slice of its data.")
    else:
        print("No class id overlap across groups -- each class's data comes from exactly one group.")

    return {"total_samples": total_samples, "total_classes": len(union_classes), "overlap": overlap}


# ---------------------------------------------------------------------------
# Class index: cheap, labels-only pass that says exactly which (rank, row)
# pairs belong to each class, so Phase 2 can fancy-index the memmaps
# directly instead of caching anything to disk.
# ---------------------------------------------------------------------------

def build_class_index(group_dir):
    """
    Reads only labels_rank*.npy (small) for this group. Returns:
      class_ids: sorted list of distinct class ids in this group
      index: dict cls_id -> {rank: ascending np.ndarray of row indices}
      local_n_by_rank: dict rank -> number of rows in that rank's shard
                       (needed to reconstruct each memmap's shape)
    """
    with open(group_dir / "meta.json") as f:
        meta = json.load(f)
    world_size = meta["world_size"]

    index = {}
    local_n_by_rank = {}
    for rank in range(world_size):
        label_path = group_dir / f"labels_rank{rank:03d}.npy"
        if not label_path.exists():
            continue
        labels = np.load(label_path)
        local_n_by_rank[rank] = labels.shape[0]

        order = np.argsort(labels, kind='stable')  # stable -> order[s:e] stays ascending per class
        sorted_labels = labels[order]
        change_points = np.nonzero(np.diff(sorted_labels))[0] + 1
        run_starts = np.concatenate(([0], change_points))
        run_ends = np.concatenate((change_points, [len(sorted_labels)]))
        for s, e in zip(run_starts, run_ends):
            cls = int(sorted_labels[s])
            index.setdefault(cls, {})[rank] = order[s:e]

    return sorted(index.keys()), index, local_n_by_rank


# ---------------------------------------------------------------------------
# Per-class fit: gather one class's rows directly from the memmaps, fit,
# pickle, discard. Runs in worker processes via a Pool initializer so the
# per-group constants (shapes/dtype/scale/etc.) are set up once per worker,
# not once per class.
# ---------------------------------------------------------------------------

_worker_state = {}


def _init_class_worker(group_dir_str, latent_shape, dtype, channel_scale,
                        local_n_by_rank, g_cfg, km_data, temp_gmm_dir_str):
    _worker_state.update(
        group_dir=Path(group_dir_str), latent_shape=latent_shape, dtype=dtype,
        np_dtype=NUMPY_DTYPE[dtype], channel_scale=channel_scale,
        local_n_by_rank=local_n_by_rank, g_cfg=g_cfg, km_data=km_data,
        temp_gmm_dir=temp_gmm_dir_str,
    )


def _fit_one_class(task):
    cls_id, rank_rows = task
    st = _worker_state

    parts = []
    for rank, row_idx in rank_rows.items():
        shard_path = st['group_dir'] / f"latents_rank{rank:03d}.dat"
        mm = np.memmap(shard_path, dtype=st['np_dtype'], mode="r",
                        shape=(st['local_n_by_rank'][rank], *st['latent_shape']))
        rows = np.asarray(mm[row_idx])  # fancy-indexing materializes just these rows into RAM
        if st['dtype'] == "int8":
            rows = rows.astype(np.float32) * st['channel_scale']
        else:
            rows = rows.astype(np.float32)
        parts.append(rows.reshape(rows.shape[0], -1))
        del mm

    X = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    # Pin this process's BLAS thread pool to 1: we're already parallel across
    # processes (one per class), so letting each one also multi-thread BLAS
    # would oversubscribe the machine.
    if threadpoolctl is not None:
        with threadpoolctl.threadpool_limits(limits=1):
            res = process_class(cls_id, X, st['g_cfg'], st['km_data'])
    else:
        res = process_class(cls_id, X, st['g_cfg'], st['km_data'])

    with open(os.path.join(st['temp_gmm_dir'], f"{cls_id}.pkl"), "wb") as f:
        pickle.dump(res, f)
    del X, res
    gc.collect()
    return cls_id


def process_group_class_by_class(group_dir, g_cfg, km_data, temp_gmm_dir, num_workers):
    """
    For one group: build the cheap class index, then fit each class's GMM
    one at a time (optionally num_workers in parallel), reading that class's
    rows directly out of the group's memmaps. No per-class or per-group
    binary cache is ever written to disk.
    """
    with open(group_dir / "meta.json") as f:
        meta = json.load(f)
    latent_shape = tuple(meta["latent_shape"])
    dtype = meta["dtype"]
    feature_dim = int(np.prod(latent_shape))

    channel_scale = None
    if dtype == "int8":
        channel_scale = np.load(group_dir / "channel_scale.npy").astype(np.float32)
        scale_shape = (1, -1) + (1,) * (len(latent_shape) - 1)
        channel_scale = channel_scale.reshape(scale_shape)

    class_ids, index, local_n_by_rank = build_class_index(group_dir)
    tasks = [(cls_id, index[cls_id]) for cls_id in class_ids]

    init_args = (str(group_dir), latent_shape, dtype, channel_scale, local_n_by_rank,
                 g_cfg, km_data, str(temp_gmm_dir))

    if num_workers == 1:
        _init_class_worker(*init_args)
        for task in tqdm(tasks, desc=f"[{group_dir.name}] class-by-class GMM fit"):
            _fit_one_class(task)
    else:
        with mp.Pool(processes=min(num_workers, len(tasks)),
                      initializer=_init_class_worker, initargs=init_args) as pool:
            list(tqdm(pool.imap_unordered(_fit_one_class, tasks),
                       total=len(tasks), desc=f"[{group_dir.name}] class-by-class GMM fit"))

    return feature_dim, class_ids


def do_gmm_clustering(latent_dir, output_dir, num_clusters=20, use_pca=False,
                       pca_dim=256, cov_type='diag', init_dir=None, num_workers=None):
    # Always check first -- cheap (labels only), and this is also the safety
    # check for group-by-group processing: valid only if no class spans more
    # than one group.
    stats = inspect_groups(latent_dir)
    if stats["overlap"] > 0:
        raise RuntimeError(
            f"{stats['overlap']} class id(s) span more than one group. This script processes one "
            f"group at a time, so a class split across groups would get fit on an incomplete slice "
            f"of its data. Fix extraction's class ranges so groups are disjoint before proceeding."
        )

    groups = discover_groups(latent_dir)
    print(f"\nProcessing {len(groups)} group(s), one class at a time within each group -- "
          f"no fp32 disk cache is built, so temp disk usage stays negligible.")

    g_cfg = {
        'num_clusters': num_clusters,
        'use_pca': use_pca,
        'pca_dim': pca_dim,
        'cov_type': cov_type,
    }

    if use_pca:
        save_dir = os.path.join(output_dir, f"{num_clusters}_{cov_type}_{pca_dim}")
    else:
        save_dir = os.path.join(output_dir, f"{num_clusters}_{cov_type}")
    os.makedirs(save_dir, exist_ok=True)
    temp_gmm_dir = os.path.join(save_dir, "temp_gmm_pkls")
    os.makedirs(temp_gmm_dir, exist_ok=True)

    km_data = None
    if init_dir:
        kmeans_path = os.path.join(init_dir, str(num_clusters), "kmeans_clusters.pkl")
        if os.path.exists(kmeans_path):
            with open(kmeans_path, "rb") as f:
                km_data = pickle.load(f)

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)

    for group_dir in groups:
        print(f"\n=== Group {group_dir.name} ===")
        process_group_class_by_class(group_dir, g_cfg, km_data, temp_gmm_dir, num_workers)
        print(f"=== Group {group_dir.name} done ===")

    # ---- Merge every class from every group into the final 9-key structure (unchanged) ----
    print("\nMerging into final 9-key structure...")
    final_dict = {
        "means": {}, "covs": {}, "weights": {}, "labels": {}, "responsibilities": {},
        "pca_components": {}, "num_components": num_clusters,
        "pca_dim": pca_dim if use_pca else None, "cov_type": cov_type,
    }
    gmm_files = sorted(os.listdir(temp_gmm_dir), key=lambda s: int(s.split('.')[0]))
    for file_name in gmm_files:
        cls_id = int(file_name.split('.')[0])
        with open(os.path.join(temp_gmm_dir, file_name), "rb") as f:
            res = pickle.load(f)
            final_dict["means"][cls_id] = res["means"]
            final_dict["covs"][cls_id] = res["covs"]
            final_dict["weights"][cls_id] = res["weights"]
            final_dict["labels"][cls_id] = res["labels"]
            final_dict["responsibilities"][cls_id] = res["responsibilities"]
            if res["pca_info"]:
                final_dict["pca_components"][cls_id] = res["pca_info"]

    if not use_pca:
        final_dict["pca_components"] = None

    final_path = os.path.join(save_dir, "gmm_clusters.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(final_dict, f)

    shutil.rmtree(temp_gmm_dir)
    print(f"✅ GMM results saved with 9 keys at: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit per-class GMMs on RAE latents extracted by extract_latents.py, "
                     "processing one group and one class at a time (no disk cache)."
    )
    parser.add_argument('--latent-dir', type=str, default='/mnt/disk1/ahlee-rae',
                         help="Parent folder holding group000/, group001/, ... (the SAME --out-dir passed "
                              "to extract_latents.py). If extraction wasn't grouped, a folder with a single "
                              "top-level meta.json also works.")
    parser.add_argument('--output-dir', type=str, default='gmm_imagenet_256',
                         help="Where to write <num_clusters>_<cov_type>/gmm_clusters.pkl.")
    parser.add_argument('--num-clusters', type=int, default=20, help="GMM components per class.")
    parser.add_argument('--use-pca', default='false', help="PCA-reduce features before fitting the GMM.")
    parser.add_argument('--pca-dim', type=int, default=1024, help="Only used if --use-pca is set.")
    parser.add_argument('--cov-type', type=str, default='diag',
                         choices=['diag', 'full', 'tied', 'spherical'], help="GaussianMixture covariance_type.")
    parser.add_argument('--init-dir', type=str, default=None,
                         help="Optional. If set, looks for <init-dir>/<num-clusters>/kmeans_clusters.pkl to "
                              "warm-start each class's GMM means from k-means centers.")
    parser.add_argument('--num-workers', type=int, default=16,
                         help="Classes fit in parallel (each worker holds one class's raw data in RAM at a "
                              "time). Default: os.cpu_count(). Set to 1 for fully sequential execution.")
    parser.add_argument('--inspect-only', action='store_true',
                         help="Only run the group/class/RAM-estimate pre-flight check (inspect_groups) and "
                              "exit -- does not fit any GMMs.")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_groups(args.latent_dir)
        sys.exit(0)

    use_pca = str(args.use_pca).lower() == "true"

    do_gmm_clustering(
        latent_dir=args.latent_dir,
        output_dir=args.output_dir,
        num_clusters=args.num_clusters,
        use_pca=use_pca,
        pca_dim=args.pca_dim,
        cov_type=args.cov_type,
        init_dir=args.init_dir,
        num_workers=args.num_workers,
    )