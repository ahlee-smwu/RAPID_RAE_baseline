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


class ShardedLatentDataset(Dataset):
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        with open(self.out_dir / "meta.json") as f:
            self.meta = json.load(f)

        self.latent_shape: Tuple[int, ...] = tuple(self.meta["latent_shape"])
        stored_dtype = self.meta["dtype"]
        np_dtype = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "int8": np.int8}[stored_dtype]
        self.is_quantized = stored_dtype == "int8"
        self.channel_scale = None
        if self.is_quantized:
            # shape (C,) -> reshape for broadcasting against (C, H, W) latents
            self.channel_scale = np.load(self.out_dir / "channel_scale.npy").reshape(-1, 1, 1).astype(np.float32)

        self.mmaps: List[np.memmap] = []
        self.labels: List[np.ndarray] = []
        self.global_index: List[np.ndarray] = []

        rank = 0
        while (self.out_dir / f"latents_rank{rank:03d}.dat").exists():
            n_in_shard = np.load(self.out_dir / f"labels_rank{rank:03d}.npy").shape[0]
            mm = np.memmap(
                self.out_dir / f"latents_rank{rank:03d}.dat",
                dtype=np_dtype,
                mode="r",
                shape=(n_in_shard, *self.latent_shape),
            )
            self.mmaps.append(mm)
            self.labels.append(np.load(self.out_dir / f"labels_rank{rank:03d}.npy"))
            self.global_index.append(np.load(self.out_dir / f"global_index_rank{rank:03d}.npy"))
            rank += 1

        if not self.mmaps:
            raise FileNotFoundError(f"No shards found under {self.out_dir}")

        shard_lens = [m.shape[0] for m in self.mmaps]
        self._offsets = np.cumsum([0] + shard_lens)  # local dataset-order offsets (not original ImageFolder order)
        self._total = int(self._offsets[-1])
        assert self._total == self.meta["total_samples"], (
            f"Shards contain {self._total} samples but meta.json expects {self.meta['total_samples']}; "
            "extraction may be incomplete."
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
        if self.is_quantized:
            latent = raw.astype(np.float32) * self.channel_scale  # dequantize -> float32, already a fresh copy
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