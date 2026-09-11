"""
evaluate_gmm_clusters.py

Evaluate a per-class GMM clustering (gmm_clusters.pkl, produced by
gmm_fit_from_rae_latents.py) against the RAE latents it was fit on, plus a
linear-probe classification eval on the same latents.

Reuses the evaluation methodology from the previously-used FFHQ/VAVAE
evaluation script almost verbatim: hard cluster assignment via Euclidean
distance to the nearest GMM mean (scipy.spatial.distance.cdist), Silhouette
/ Davies-Bouldin / Calinski-Harabasz on that assignment, effective number of
active clusters (Neff) and responsibility entropy from the fitted
`responsibilities`, kNN cluster-preservation, and per-class plots (an
inter-cluster distance heatmap and a cluster-weight bar chart).

Because assignment here is pure Euclidean-to-mean (not Mahalanobis), it
never touches the GMM's covariances -- so, unlike a Mahalanobis-based
assignment, this works identically whether or not --use-pca was used when
fitting (process_class() always stores `means` inverse-transformed back to
the original D-dim space, so X and means are always in the same space here
regardless of PCA). The tradeoff: this ignores each cluster's fitted shape
(covariance) when assigning points, unlike Mahalanobis distance would.

WHAT CHANGED vs. the reference script: latent loading. The reference
streamed the ENTIRE dataset through ImgLatentDataset + DataLoader into a
per-class .bin disk cache before evaluating (stream_latents_to_disk /
load_class_latents) -- exactly the disk-heavy caching pattern this
conversation already removed from the GMM-fitting script for the same
500 GB-constrained /mnt/disk1/ahlee-rae setup. Since evaluation only needs a
bounded sample per class (not the whole dataset), this version instead:
    <latent_dir>/group{NNN}/meta.json
    <latent_dir>/group{NNN}/latents_rank{R:03d}.dat
    <latent_dir>/group{NNN}/labels_rank{R:03d}.npy
    <latent_dir>/group{NNN}/channel_scale.npy   (only if dtype == 'int8')
builds a cheap labels-only class index (same trick as
gmm_fit_from_rae_latents.py's build_class_index) and fancy-indexes directly
into the memmaps for just the sampled rows. No temp disk cache, nothing to
clean up in a `finally` block.

NEW: evaluate_linear_probe() -- fits a single linear layer (softmax /
logistic regression via a few epochs of mini-batch Adam in PyTorch) on the
same sampled latents to predict class labels, reporting held-out top-1 /
top-5 accuracy -- how linearly separable class identity is in the raw
latent space, independent of the GMM clustering above.

WHY THIS VERSION IS FAST: contiguous slices, not scattered fancy-indexing
---------------------------------------------------------------------------
extract_latents.py writes each rank's shard rows in the exact order
DistributedSampler(shuffle=False) enumerates them, and that sampler just
strides through an already class-contiguous ImageFolder ordering (optionally
narrowed to one --classes-per-group range). The consequence: within any one
rank's shard, a given class's rows always form a SINGLE CONTIGUOUS BLOCK,
never scattered positions. build_class_index() detects this (falling back
safely to a plain index array if it's ever NOT contiguous) and stores either
a (start, stop) pair or a raw index array per (class, rank). fetch_class_samples()
then does a real slice `mm[start:stop]` whenever possible instead of fancy
indexing `mm[row_idx]` -- a real slice is one sequential disk read; fancy
indexing an array of individual positions is effectively random access, even
when those positions happen to be consecutive integers. This distinction is
what separates ~7s/class from a small fraction of that.
Both the per-class sampling pass and the per-class metric/plot pass are also
parallelized across --num-workers processes (classes are independent), since
CPU/RAM/GPU headroom is available.
---------------------------------------------------------------------------

Usage:
    python evaluate_gmm_clusters.py
    python evaluate_gmm_clusters.py --latent-dir /mnt/disk1/ahlee-rae \
        --pkl-path gmm_imagenet_256/20_diag/gmm_clusters.pkl --no-plots --num-workers 32
"""

import os
import json
import pickle
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe backend -- no $DISPLAY needed on a training server
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

# Matches extract_latents.py's NUMPY_DTYPE table exactly (bf16 has no native
# numpy dtype, so it's stored on disk as fp32 there too).
NUMPY_DTYPE = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "int8": np.int8}


# ---------------------------------------------------------------------------
# 1. Latent access: discover groups, build a cheap labels-only class index,
#    and fetch a bounded per-class sample directly from the memmaps.
#    (Same design as gmm_fit_from_rae_latents.py -- no disk cache.)
# ---------------------------------------------------------------------------

def discover_groups(latent_dir):
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


def _compact_run(idx_array):
    """
    idx_array is ascending (comes from a stable argsort). If it's a REAL
    contiguous range of integers -- which it always is for this dataset's
    extraction order, see the module docstring -- return (start, stop) so
    the caller can use a real slice. Otherwise return the array as-is for a
    (slower, but correct) fancy-index fallback.
    """
    if len(idx_array) == 0:
        return (0, 0)
    start, stop = int(idx_array[0]), int(idx_array[-1]) + 1
    if stop - start == len(idx_array):
        return (start, stop)
    return idx_array


def build_class_index(group_dir):
    """
    Labels-only pass. Returns:
      index: cls -> {rank: (start, stop) if contiguous, else raw row-idx array}
      local_n_by_rank: rank -> number of rows in that rank's shard
      scattered: list of (rank, cls) pairs that hit the fancy-index fallback
                 (i.e. weren't a real contiguous range) -- normally empty.
    """
    with open(group_dir / "meta.json") as f:
        meta = json.load(f)
    world_size = meta["world_size"]

    index = {}
    local_n_by_rank = {}
    scattered = []
    for rank in range(world_size):
        label_path = group_dir / f"labels_rank{rank:03d}.npy"
        if not label_path.exists():
            continue
        labels = np.load(label_path)
        local_n_by_rank[rank] = labels.shape[0]

        order = np.argsort(labels, kind='stable')
        sorted_labels = labels[order]
        change_points = np.nonzero(np.diff(sorted_labels))[0] + 1
        run_starts = np.concatenate(([0], change_points))
        run_ends = np.concatenate((change_points, [len(sorted_labels)]))
        for s, e in zip(run_starts, run_ends):
            cls = int(sorted_labels[s])
            compact = _compact_run(order[s:e])
            if not isinstance(compact, tuple):
                scattered.append((rank, cls))
            index.setdefault(cls, {})[rank] = compact

    return index, local_n_by_rank, scattered


def build_global_class_index(latent_dir):
    """
    Merges build_class_index() across every discovered group. Verifies
    classes don't span more than one group (true for this dataset's
    --classes-per-group extraction layout).
    """
    groups = discover_groups(latent_dir)
    class_group, class_rows, group_meta = {}, {}, {}
    total_stats = {"contig": 0, "scattered": 0}
    scattered_detail = []  # (group_dir.name, rank, cls)

    for group_dir in groups:
        with open(group_dir / "meta.json") as f:
            meta = json.load(f)
        index, local_n_by_rank, scattered = build_class_index(group_dir)
        n_runs = sum(len(v) for v in index.values())
        total_stats["contig"] += n_runs - len(scattered)
        total_stats["scattered"] += len(scattered)
        scattered_detail += [(group_dir.name, rank, cls) for rank, cls in scattered]

        channel_scale = None
        if meta["dtype"] == "int8":
            latent_shape = tuple(meta["latent_shape"])
            channel_scale = np.load(group_dir / "channel_scale.npy").astype(np.float32)
            scale_shape = (1, -1) + (1,) * (len(latent_shape) - 1)
            channel_scale = channel_scale.reshape(scale_shape)
        group_meta[group_dir] = (meta, local_n_by_rank, channel_scale)

        for cls, rank_rows in index.items():
            if cls in class_group and class_group[cls] != group_dir:
                raise RuntimeError(
                    f"Class {cls} appears in more than one group ({class_group[cls].name} and "
                    f"{group_dir.name}) -- this script assumes disjoint per-group class ranges."
                )
            class_group[cls] = group_dir
            class_rows[cls] = rank_rows

    return sorted(class_group.keys()), class_group, class_rows, group_meta, total_stats, scattered_detail


def _run_len(v):
    return (v[1] - v[0]) if isinstance(v, tuple) else len(v)


def fetch_class_samples(cls, class_group, class_rows, group_meta, max_per_class, rng):
    """
    Fetch up to max_per_class rows for one class directly from its group's
    memmap shard(s). Uses a real slice (mm[start:stop], one sequential disk
    read) whenever the (class, rank) run is contiguous -- which it always
    is for this dataset (see module docstring) -- and falls back to fancy
    indexing only for the rare non-contiguous case. When more rows are
    available than max_per_class, takes a randomly-offset CONTIGUOUS window
    from each rank's run (not scattered picks), so subsampling never turns a
    fast sequential read back into a slow scattered one.
    """
    group_dir = class_group[cls]
    meta, local_n_by_rank, channel_scale = group_meta[group_dir]
    latent_shape = tuple(meta["latent_shape"])
    dtype = meta["dtype"]
    np_dtype = NUMPY_DTYPE[dtype]

    rank_rows = class_rows[cls]
    total_available = sum(_run_len(v) for v in rank_rows.values())

    if total_available > max_per_class:
        ranks = list(rank_rows.keys())
        remaining = max_per_class
        rank_plan = {}
        for i, r in enumerate(ranks):
            avail = _run_len(rank_rows[r])
            take = remaining if i == len(ranks) - 1 else min(avail, round(max_per_class * avail / total_available))
            take = max(0, min(take, avail, remaining))
            remaining -= take
            if take == 0:
                continue
            v = rank_rows[r]
            if isinstance(v, tuple):
                start, stop = v
                offset = int(rng.integers(0, (stop - start) - take + 1)) if (stop - start) > take else 0
                rank_plan[r] = (start + offset, start + offset + take)
            else:
                chosen = rng.choice(len(v), size=take, replace=False)
                rank_plan[r] = np.sort(v[chosen])
    else:
        rank_plan = rank_rows

    parts = []
    for rank, spec in rank_plan.items():
        shard_path = group_dir / f"latents_rank{rank:03d}.dat"
        mm = np.memmap(shard_path, dtype=np_dtype, mode="r",
                        shape=(local_n_by_rank[rank], *latent_shape))
        rows = np.asarray(mm[spec[0]:spec[1]]) if isinstance(spec, tuple) else np.asarray(mm[spec])
        rows = rows.astype(np.float32) * channel_scale if dtype == "int8" else rows.astype(np.float32)
        parts.append(rows.reshape(rows.shape[0], -1))
        del mm

    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


_sample_worker_state = {}


def _init_sample_worker(class_group, class_rows, group_meta, max_per_class, seed):
    _sample_worker_state.update(class_group=class_group, class_rows=class_rows, group_meta=group_meta,
                                 max_per_class=max_per_class, rng=np.random.default_rng(seed))


def _fetch_one_class(cls):
    st = _sample_worker_state
    X = fetch_class_samples(cls, st['class_group'], st['class_rows'], st['group_meta'], st['max_per_class'], st['rng'])
    return cls, X


def collect_sampled_latents_from_groups(latent_dir, max_per_class=600, max_global=60000, seed=0, num_workers=None):
    """
    Returns (global_latents, global_labels, class_latents): class_latents[cls]
    holds up to max_per_class samples for that class; the global arrays pool
    up to max_global samples roughly evenly across classes (for the linear
    probe), subsampled from the already-fetched per-class data. Classes are
    fetched across --num-workers processes since they're independent.
    """
    class_ids, class_group, class_rows, group_meta, stats, scattered_detail = build_global_class_index(latent_dir)
    print(f"[Index] {stats['contig']} (class, rank) run(s) are contiguous (fast slice read), "
          f"{stats['scattered']} need fancy-index fallback.")
    if scattered_detail:
        shown = scattered_detail[:20]
        print(f"[Index] scattered (group, rank, class): {shown}"
              + (f" ... and {len(scattered_detail) - 20} more" if len(scattered_detail) > 20 else ""))
        print("[Index] this just means those classes fall back to the slower (but still correct) "
              "fancy-index path -- likely candidates are classes near an interrupted/recovered "
              "extraction boundary (see recover_partial_extraction.py) or a partial rank shard; "
              "safe to ignore unless the count is large.")

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)

    class_latents = {}
    if num_workers == 1:
        rng = np.random.default_rng(seed)
        for cls in tqdm(class_ids, desc="Sampling per-class latents"):
            class_latents[cls] = fetch_class_samples(cls, class_group, class_rows, group_meta, max_per_class, rng)
    else:
        with mp.Pool(processes=min(num_workers, len(class_ids)), initializer=_init_sample_worker,
                      initargs=(class_group, class_rows, group_meta, max_per_class, seed)) as pool:
            for cls, X in tqdm(pool.imap_unordered(_fetch_one_class, class_ids), total=len(class_ids),
                                desc="Sampling per-class latents"):
                class_latents[cls] = X

    per_class_global_cap = max(1, max_global // len(class_ids))
    rng = np.random.default_rng(seed)
    global_parts, global_label_parts = [], []
    for cls in class_ids:
        X = class_latents[cls]
        n_take = min(per_class_global_cap, len(X))
        sel = rng.choice(len(X), size=n_take, replace=False) if n_take < len(X) else np.arange(len(X))
        global_parts.append(X[sel])
        global_label_parts.append(np.full(n_take, cls, dtype=np.int64))

    global_latents = np.concatenate(global_parts)[:max_global]
    global_labels = np.concatenate(global_label_parts)[:max_global]

    print(f"[Sampling] global={len(global_latents)}, classes={len(class_latents)}, "
          f"cap/class={max_per_class} (global subsample/class≈{per_class_global_cap})")
    return global_latents, global_labels, class_latents


# ---------------------------------------------------------------------------
# 2. Metric helpers (unchanged math from the reference script)
# ---------------------------------------------------------------------------

def compute_neff_from_responsibility(R, eps=1e-12):
    p_k = R.mean(axis=0)
    H = -np.sum(p_k * np.log(p_k + eps))
    return np.exp(H), p_k


def compute_responsibility_entropy(R, eps=1e-12):
    H_i = -np.sum(R * np.log(R + eps), axis=1)
    return H_i.mean(), np.median(H_i)


def compute_knn_preservation(X, cluster_assigns, k=10, max_samples=5000):
    N = len(X)
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        X = X[idx]
        cluster_assigns = cluster_assigns[idx]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, knn_idx = nbrs.kneighbors(X)

    preserve_ratios = []
    for i in range(len(X)):
        true_neighbors = knn_idx[i, 1:]
        same_cluster = np.where(cluster_assigns == cluster_assigns[i])[0]
        overlap = np.intersect1d(true_neighbors, same_cluster)
        preserve_ratios.append(len(overlap) / k)

    return np.mean(preserve_ratios)


# ---------------------------------------------------------------------------
# 3. Class-wise GMM evaluation + plots (adapted from the reference: instead
#    of lazily loading each class from a temp disk cache, class_latents is
#    already the in-memory dict from collect_sampled_latents_from_groups).
# ---------------------------------------------------------------------------

def _evaluate_one_class(task):
    """Per-class body of evaluate_gmm_performance, runnable in a worker process."""
    cls, X = task
    st = _eval_worker_state
    gmm_params, save_path = st['gmm_params'], st['save_path']
    knn_k, make_plots, metric_sample = st['knn_k'], st['make_plots'], st['metric_sample']
    rng = np.random.default_rng(st['seed'] + cls)  # per-class offset -> independent across workers
    responsibilities = gmm_params.get('responsibilities') or {}

    if cls not in gmm_params['means'] or X is None or len(X) == 0:
        return None

    means = gmm_params['means'][cls]
    weights = gmm_params['weights'][cls]

    dists = cdist(X, means, metric='euclidean')
    cluster_assigns = np.argmin(dists, axis=1)

    n_sub = min(metric_sample, len(X))
    s_idx = rng.choice(len(X), n_sub, replace=False) if n_sub < len(X) else np.arange(len(X))

    if n_sub >= 2 and len(np.unique(cluster_assigns[s_idx])) >= 2:
        sil = silhouette_score(X[s_idx], cluster_assigns[s_idx])
        db_idx = davies_bouldin_score(X[s_idx], cluster_assigns[s_idx])
        ch_idx = calinski_harabasz_score(X[s_idx], cluster_assigns[s_idx])
    else:
        sil = db_idx = ch_idx = np.nan

    if cls in responsibilities and len(responsibilities[cls]) == len(X):
        R = responsibilities[cls]
        Neff, p_k = compute_neff_from_responsibility(R)
        Neff_norm = Neff / len(p_k)
        H_mean, H_median = compute_responsibility_entropy(R)
    else:
        Neff, Neff_norm, H_mean, H_median = np.nan, np.nan, np.nan, np.nan

    k_eff = min(knn_k, n_sub - 1)
    knn_preserve = compute_knn_preservation(X[s_idx], cluster_assigns[s_idx], k=k_eff) if k_eff >= 1 else np.nan

    if make_plots:
        plt.figure(figsize=(10, 8))
        inter_dist = cdist(means, means, metric='euclidean')
        sns.heatmap(inter_dist, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title(f"Class {cls}: Inter-cluster Distance Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"class_{cls}_dist_heatmap.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(range(len(weights)), weights, color='royalblue', alpha=0.7)
        plt.xticks(range(len(weights)))
        plt.title(f"Class {cls}: Cluster Mixing Coefficients (Weights)")
        plt.xlabel("Cluster ID")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"class_{cls}_weights.png"))
        plt.close()

    return dict(Class=cls, Silhouette=sil, DB_Index=db_idx, CH_Index=ch_idx,
                Avg_Dist=np.min(dists, axis=1).mean(), Neff=Neff, Neff_norm=Neff_norm,
                RespEnt_mean=H_mean, RespEnt_median=H_median, KNN_preserve=knn_preserve)


_eval_worker_state = {}


def _init_eval_worker(gmm_params, save_path, knn_k, make_plots, metric_sample, seed):
    _eval_worker_state.update(gmm_params=gmm_params, save_path=save_path, knn_k=knn_k,
                               make_plots=make_plots, metric_sample=metric_sample, seed=seed)


def evaluate_gmm_performance(class_latents, gmm_params, save_path, knn_k=10,
                              make_plots=True, metric_sample=10000, seed=0, num_workers=None):
    os.makedirs(save_path, exist_ok=True)
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)

    tasks = list(class_latents.items())
    report = []
    if num_workers == 1:
        _init_eval_worker(gmm_params, save_path, knn_k, make_plots, metric_sample, seed)
        for task in tqdm(tasks, desc="Evaluating classes"):
            row = _evaluate_one_class(task)
            if row is not None:
                report.append(row)
    else:
        with mp.Pool(processes=min(num_workers, len(tasks)), initializer=_init_eval_worker,
                      initargs=(gmm_params, save_path, knn_k, make_plots, metric_sample, seed)) as pool:
            for row in tqdm(pool.imap_unordered(_evaluate_one_class, tasks), total=len(tasks),
                             desc="Evaluating classes"):
                if row is not None:
                    report.append(row)
    report.sort(key=lambda r: r['Class'])

    print("\n" + "=" * 95)
    print(f"{'Class':<6} | {'Silh':<8} | {'DBI':<8} | {'CH':<8} | {'Dist':<8} | "
          f"{'Neff':<8} | {'N/K':<6} | {'Ent':<8} | {'kNN':<6}")
    print("-" * 95)
    for r in report:
        print(f"{r['Class']:<6} | {r['Silhouette']:>8.4f} | {r['DB_Index']:>8.3f} | {r['CH_Index']:>8.1f} | "
              f"{r['Avg_Dist']:>8.2f} | {r['Neff']:>8.2f} | {r['Neff_norm']:>6.3f} | "
              f"{r['RespEnt_mean']:>8.3f} | {r['KNN_preserve']:>6.3f}")
    print("=" * 95)
    return report


# ---------------------------------------------------------------------------
# 4. NEW: linear probe -- one linear layer on frozen latents predicting
#    class labels, held-out top-1/top-5 accuracy.
# ---------------------------------------------------------------------------

def evaluate_linear_probe(latents, labels, train_frac=0.8, epochs=20, lr=1e-2,
                           batch_size=1024, weight_decay=1e-4, seed=0, device=None):
    print("\n📈 Linear Probe Evaluation")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(seed)
    N, D = latents.shape
    num_classes = int(labels.max()) + 1

    perm = rng.permutation(N)
    n_train = int(N * train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("[skip] not enough samples to form a train/test split for the linear probe.")
        return None

    mu = latents[train_idx].mean(axis=0, keepdims=True)
    sigma = latents[train_idx].std(axis=0, keepdims=True) + 1e-6

    X_train = torch.from_numpy(((latents[train_idx] - mu) / sigma).astype(np.float32)).to(device)
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64)).to(device)
    X_test = torch.from_numpy(((latents[test_idx] - mu) / sigma).astype(np.float32)).to(device)
    y_test = torch.from_numpy(labels[test_idx].astype(np.int64)).to(device)

    probe = torch.nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    n_train_samples = X_train.shape[0]
    log_every = max(1, epochs // 5)
    for epoch in range(epochs):
        probe.train()
        epoch_perm = torch.randperm(n_train_samples, device=device)
        total_loss = 0.0
        for start in range(0, n_train_samples, batch_size):
            batch_idx = epoch_perm[start:start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1:>3d}/{epochs} - train loss: {total_loss / n_train_samples:.4f}")

    probe.eval()
    with torch.no_grad():
        logits = probe(X_test)
        top1 = (logits.argmax(dim=1) == y_test).float().mean().item()
        k5 = min(5, num_classes)
        top5_idx = torch.topk(logits, k=k5, dim=1).indices
        top5 = (top5_idx == y_test.unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"Linear probe top-1 accuracy: {top1:.4f}")
    if k5 > 1:
        print(f"Linear probe top-5 accuracy: {top5:.4f}")
    print(f"(train={n_train_samples}, test={len(test_idx)}, dim={D}, classes={num_classes}, device={device})")
    return dict(top1=top1, top5=top5)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM cluster evaluation + linear probe on RAE latents")
    parser.add_argument('--latent-dir', type=str, default='/mnt/disk1/ahlee-rae',
                         help="Parent folder holding group000/, group001/, ... (extract_latents.py's --out-dir).")
    parser.add_argument('--pkl-path', type=str, default='gmm_imagenet_256/20_diag/gmm_clusters_save.pkl',
                         help="Path to the gmm_clusters.pkl produced by gmm_fit_from_rae_latents.py.")
    parser.add_argument('--output-dir', type=str, default=None,
                         help="Where to save per-class plots. Default: same directory as --pkl-path.")
    parser.add_argument('--max-per-class', type=int, default=300, help="Samples per class for GMM evaluation.")
    parser.add_argument('--max-global', type=int, default=30000, help="Pooled sample size for the linear probe.")
    parser.add_argument('--knn-k', type=int, default=10, help="k for kNN cluster-preservation score.")
    parser.add_argument('--no-plots', action='store_true',
                         help="Skip per-class PNG plots (2 per class -- with 1000 classes that's ~2000 files).")
    parser.add_argument('--lp-epochs', type=int, default=20, help="Linear probe training epochs.")
    parser.add_argument('--lp-lr', type=float, default=1e-2, help="Linear probe learning rate (Adam).")
    parser.add_argument('--lp-batch-size', type=int, default=1024, help="Linear probe mini-batch size.")
    parser.add_argument('--lp-train-frac', type=float, default=0.8, help="Train fraction for the linear probe split.")
    parser.add_argument('--num-workers', type=int, default=16,
                         help="Worker processes for BOTH per-class sampling and per-class GMM evaluation "
                              "(classes are independent, so both parallelize across processes). "
                              "Default: os.cpu_count(). Set to 1 for fully sequential execution.")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(
            f"--pkl-path {args.pkl_path!r} does not exist (cwd={os.getcwd()}). Check the path is relative "
            f"to where you're running this script from, or pass an absolute path."
        )
    with open(args.pkl_path, "rb") as f:
        gmm_params = pickle.load(f)

    cov_type = gmm_params.get('cov_type', 'diag')
    use_pca = gmm_params.get('pca_components') is not None
    print(f"[GMM] classes={len(gmm_params['means'])}, cov_type={cov_type}, PCA={use_pca} "
          f"(assignment here is Euclidean-to-mean, so cov_type/PCA don't affect it -- "
          f"see the module docstring)")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.pkl_path))

    latents, labels, class_latents = collect_sampled_latents_from_groups(
        args.latent_dir, max_per_class=args.max_per_class, max_global=args.max_global,
        seed=args.seed, num_workers=args.num_workers)

    evaluate_gmm_performance(class_latents, gmm_params, output_dir, knn_k=args.knn_k,
                              make_plots=not args.no_plots, seed=args.seed, num_workers=args.num_workers)

    evaluate_linear_probe(latents, labels, train_frac=args.lp_train_frac, epochs=args.lp_epochs,
                           lr=args.lp_lr, batch_size=args.lp_batch_size, seed=args.seed)