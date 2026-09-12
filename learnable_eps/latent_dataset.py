"""
latent_dataset.py

Read-side counterpart to extract_latents.py. Presents the sharded memmap
latents as a single indexable torch Dataset with O(1) random access and no
extra RAM beyond the OS page cache (memmaps are lazily paged in).

__getitem__ always returns float32 tensors, regardless of on-disk storage
dtype (fp16 / int8 / fp32) -- this matches what train.py's online
`rae.encode(images)` call actually produces (fp32, uncast). If you need a
lower-precision tensor for training, cast it yourself after `.to(device)`
(e.g. `z.to(device).to(torch.bfloat16)`), inside whatever autocast/precision
scheme your training loop already uses -- don't rely on the loader to hand
you anything other than fp32.

Usage:
    ds = ShardedLatentDataset("/fast_local_disk/imagenet_latents")
    latent, label = ds[12345]
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8,
                         persistent_workers=True, prefetch_factor=4)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def discover_groups(latent_dir) -> List[Path]:
    """Directories that each hold one meta.json plus their own rank shards.

    Handles both extract_z.py layouts transparently:
      grouped   -- <out_dir>/group000/, group001/, ...   (--classes-per-group)
      ungrouped -- <out_dir>/ itself

    Same discovery rule as gmm_fit.py's discover_groups, so the loader and the
    GMM fit always see the same set of shards.
    """
    latent_dir = Path(latent_dir)
    group_metas = sorted(latent_dir.glob("group*/meta.json"))
    if group_metas:
        return [m.parent for m in group_metas]
    if (latent_dir / "meta.json").exists():
        return [latent_dir]
    raise FileNotFoundError(
        f"No meta.json found directly under {latent_dir} or under {latent_dir}/group*/ -- "
        f"is this really an extract_z.py --out-dir?"
    )


_NP_DTYPE = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "int8": np.int8}


class ShardedLatentDataset(Dataset):
    """Flat, randomly-indexable view over every shard of an extraction.

    Class-group extractions write one independent sub-directory per group, each
    with its own meta.json, rank shards and (for int8) channel_scale.npy. They
    are concatenated here into a single dataset, so a group boundary is
    invisible to the sampler and each epoch still shuffles across all classes.

    Labels are stored as GLOBAL class ids: extract_z.py wraps the full
    ImageFolder in a Subset, which passes the underlying target through
    unchanged, so a grouped extraction still records 0..999 and needs no
    remapping.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.group_dirs = discover_groups(self.out_dir)

        self.mmaps: List[np.memmap] = []
        self.labels: List[np.ndarray] = []
        self.global_index: List[np.ndarray] = []
        self.shard_scale: List[np.ndarray | None] = []   # per shard: int8 dequant scale or None

        self.latent_shape: Tuple[int, ...] | None = None
        stored_dtype = None
        self.meta = None
        expected_total = 0

        for gdir in self.group_dirs:
            with open(gdir / "meta.json") as f:
                meta = json.load(f)
            if self.meta is None:
                self.meta = meta

            g_shape = tuple(meta["latent_shape"])
            g_dtype = meta["dtype"]
            if self.latent_shape is None:
                self.latent_shape, stored_dtype = g_shape, g_dtype
            elif g_shape != self.latent_shape or g_dtype != stored_dtype:
                # Mixing shapes or dtypes would silently corrupt the batch.
                raise ValueError(
                    f"{gdir.name}: latent_shape/dtype {g_shape}/{g_dtype} does not match "
                    f"{self.latent_shape}/{stored_dtype} from {self.group_dirs[0].name}."
                )

            np_dtype = _NP_DTYPE[g_dtype]
            scale = None
            if g_dtype == "int8":
                # Calibrated per group, so it must be applied per group.
                scale = np.load(gdir / "channel_scale.npy").reshape(-1, 1, 1).astype(np.float32)

            n_before = len(self.mmaps)
            # Iterate the rank count meta.json declares rather than probing for
            # .dat files: extract_z.py preallocates latents_rank{R}.dat at full
            # size BEFORE its encode loop and writes labels/global_index only
            # after the loop finishes, so a .dat on its own proves nothing --
            # a rank that died early leaves a full-size, zero-filled one.
            for r in range(int(meta["world_size"])):
                dat = gdir / f"latents_rank{r:03d}.dat"
                lab = gdir / f"labels_rank{r:03d}.npy"
                gix = gdir / f"global_index_rank{r:03d}.npy"
                missing = [f.name for f in (dat, lab, gix) if not f.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"{gdir}: rank {r} of {meta['world_size']} is incomplete "
                        f"(missing {', '.join(missing)}).\n"
                        f"extract_z.py writes labels_rank{r:03d}.npy only after that rank "
                        f"finishes, so this shard never completed and its .dat is "
                        f"zero-filled. Training on it would feed the model empty latents.\n"
                        f"Run: python learnable_eps/check_latents.py --latent-path {self.out_dir}"
                    )
                labels = np.load(lab)
                self.mmaps.append(np.memmap(
                    dat, dtype=np_dtype, mode="r",
                    shape=(labels.shape[0], *self.latent_shape),
                ))
                self.labels.append(labels)
                self.global_index.append(np.load(gix))
                self.shard_scale.append(scale)

            if len(self.mmaps) == n_before:
                raise FileNotFoundError(f"No latents_rank*.dat shards under {gdir}")

            got = sum(m.shape[0] for m in self.mmaps[n_before:])
            if got != meta["total_samples"]:
                raise ValueError(
                    f"{gdir.name}: shards hold {got} samples but meta.json expects "
                    f"{meta['total_samples']}; extraction may be incomplete."
                )
            expected_total += int(meta["total_samples"])

        self.is_quantized = stored_dtype == "int8"
        # Kept for backward compatibility with the single-group layout.
        self.channel_scale = self.shard_scale[0] if self.shard_scale else None

        shard_lens = [m.shape[0] for m in self.mmaps]
        self._offsets = np.cumsum([0] + shard_lens)
        self._total = int(self._offsets[-1])
        assert self._total == expected_total, (
            f"Concatenated {self._total} samples but group metas sum to {expected_total}."
        )

    def describe(self) -> str:
        return (
            f"{self._total} samples, {len(self.mmaps)} shards across "
            f"{len(self.group_dirs)} group(s) [{', '.join(g.name for g in self.group_dirs)}], "
            f"latent_shape={self.latent_shape}, dtype={self.meta['dtype']}"
        )

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_id = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - self._offsets[shard_id]
        return shard_id, int(local_idx)

    def __getitem__(self, idx: int):
        shard_id, local_idx = self._locate(idx)
        raw = self.mmaps[shard_id][local_idx]
        label = int(self.labels[shard_id][local_idx])
        scale = self.shard_scale[shard_id]
        if scale is not None:
            latent = raw.astype(np.float32) * scale  # dequantize -> float32, fresh copy
        else:
            # Always upcast to float32 here, regardless of on-disk storage dtype (fp16/bf16-as-fp32/fp32).
            # This matches what train.py's online `rae.encode(images)` actually returns (fp32, since that
            # call isn't wrapped in autocast) -- storage dtype is purely a disk-space optimization and must
            # not leak into the dtype the model sees. `.astype(..., copy=True)` also produces a writable
            # array, avoiding the "not writable" UserWarning from wrapping a read-only memmap view directly.
            latent = raw.astype(np.float32, copy=True)
        return torch.from_numpy(latent), label

    def original_imagefolder_index(self, idx: int) -> int:
        """Map a dataset-order idx back to the index in the original ImageFolder dataset."""
        shard_id, local_idx = self._locate(idx)
        return int(self.global_index[shard_id][local_idx])


def prepare_latent_dataloader(
    latent_dir,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    seed: int = 0,
):
    """
    Drop-in replacement for `prepare_dataloader(data_path, batch_size, num_workers,
    rank, world_size, transform=...)` in train.py, but reading pre-extracted latents
    instead of decoding+encoding raw images every step.

    Mirrors the same shape of return value -- (loader, sampler) -- and the same
    per-epoch reshuffling behavior (`sampler.set_epoch(epoch)` still works), so the
    only thing that needs to change at the call site in train.py is which function
    is called and what the loader yields (latents instead of images).

    Note: extraction shards data across ranks in a fixed, non-shuffled way (see
    extract_latents.py), but that only determines which *file* a sample lives in --
    ShardedLatentDataset exposes all shards as one flat, randomly-indexable dataset,
    so a fresh shuffled DistributedSampler here still shuffles across the *entire*
    dataset and across ranks every epoch, exactly like the original image loader did.
    """
    dataset = ShardedLatentDataset(latent_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    return loader, sampler