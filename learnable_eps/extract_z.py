# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
extract_latents.py

Run the FROZEN RAE encoder (same instantiation/checkpoint-loading logic as the
stage-1 training script) over an ImageFolder-style ImageNet dataset and store
the resulting latents as memory-mapped binary shards.

Why memmap shards instead of one-file-per-image?
  - 1.2M separate .pt/.npy files means 1.2M inodes -> filesystem metadata
    overhead, terrible random-read throughput, and (as you found with the
    GMM/DiT pipeline) it turns any HDD/NFS-backed storage into an IO-starved
    bottleneck.
  - A handful of large contiguous binary files (one per rank) means:
      * O(1) random access via a precomputed index (no per-sample fs lookup)
      * OS page cache can actually hold hot regions in RAM
      * writes are sequential per rank -> fast, no lock contention
      * trivial to compute exact size ahead of time and preallocate on disk

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=8 extract_latents.py \
        --config configs/stage1/training/DINOv2-B_decXL.yaml \
        --data-path /path/to/imagenet/train \
        --out-dir /fast_local_disk/imagenet_latents \
        --image-size 256 \
        --batch-size 512
    # (--dtype defaults to fp32 now: bit-exact latents, no quantization risk.
    #  Pass --dtype fp16 or --dtype int8 only if you need to trade fidelity for disk space.)

Extract in class-groups to cap peak disk usage (e.g. 250 of 1000 classes at a
time -> output goes to <out-dir>/group000/, group001/, ... independently):
    torchrun --standalone --nproc_per_node=8 extract_latents.py \
        --config configs/stage1/training/DINOv2-B_decXL.yaml \
        --data-path /path/to/imagenet/train \
        --out-dir /fast_local_disk/imagenet_latents \
        --image-size 256 --batch-size 512 \
        --classes-per-group 250 --group-idx 0   # then repeat with --group-idx 1, 2, 3

Watch live progress from another terminal/tmux pane while the above is running
(same file, no GPU/torchrun needed for this mode):
    python extract_latents.py --monitor --out-dir /fast_local_disk/imagenet_latents --refresh 5

Or, one command that does both automatically (auto-creates a tmux session
with extraction + live monitor in split panes and attaches you to it -- plain
`python`, not torchrun, since this re-invokes torchrun itself):
    python extract_latents.py --tmux --nproc-per-node 8 \
        --config configs/stage1/training/DINOv2-B_decXL.yaml \
        --data-path /path/to/imagenet/train \
        --out-dir /fast_local_disk/imagenet_latents \
        --image-size 256 --batch-size 512

Notes:
  - Encoder settings (instantiation, checkpoint keys, eval/no-grad) are kept
    identical to train_stage1.py so the latents match what the decoder was
    trained against.
  - Center-crop (not random-crop) is used here since this is a one-shot,
    deterministic feature-extraction pass, not augmentation for training.
  - IMPORTANT: this calls rae.encode(images) -- the same public API used in
    train.py's stage-2 loop (`z = rae.encode(images)`) -- rather than calling
    the raw `rae.encoder` submodule directly. If RAE.encode() does anything
    beyond the bare encoder forward (scaling, normalization, etc.), calling
    the submodule directly would silently produce latents that don't match
    what stage-2 training expects.
  - train.py's on-the-fly encode call is NOT wrapped in autocast (see the
    `# TODO: wrap this in autocast?` comment there), so by default this
    script also runs encode() in full precision to match exactly. Storage
    dtype (--dtype) is a separate, independent choice from compute precision
    (--compute-dtype) -- see below.
  - PREPROCESSING MUST MATCH stage-2's `stage2_transform`, not stage-1's
    `stage1_transform`. train_stage1.py's Resize+RandomCrop pipeline is
    decoder-training augmentation (random, not reproducible) and is the wrong
    thing to reuse here. What actually needs to match is what train.py feeds
    into `rae.encode(images)`: `center_crop_arr(image, image_size)` (imported
    from utils.train_utils, same as train.py -- NOT torchvision's generic
    Resize+CenterCrop, which uses a different resampling path and will
    silently produce slightly different pixels/latents). RandomHorizontalFlip
    is dropped here since it's a random per-step augmentation that can't be
    baked into a fixed cache; if you rely on flip augmentation in stage-2
    training, either extract a second flipped copy of every latent (2x
    storage) or accept giving up flip augmentation when training from cache.
"""

from __future__ import annotations

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(os.path.join(root_dir, 'stage1'))

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config
from utils.dist_utils import setup_distributed, cleanup_distributed
from utils.train_utils import parse_configs, center_crop_arr  # same center_crop_arr used by train.py's stage2_transform
from stage1 import RAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frozen RAE encoder latents to memmap shards.")
    p.add_argument("--config", type=str, default=None, help="Same YAML used for stage-1 training. Required unless --monitor.")
    p.add_argument("--data-path", type=Path, default=None, help="Required unless --monitor.")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Optional. NOT needed in the normal case: train_stage1.py freezes the encoder for "
                         "the entire run (rae.encoder.requires_grad_(False), never updated), so the encoder "
                         "weights produced by instantiate_from_config(rae_config) alone already match what "
                         "any stage-1 checkpoint would contain for the encoder. Only pass this if your config "
                         "does NOT bake in pretrained encoder weights on instantiation and you need to pull "
                         "them from a checkpoint instead.")
    p.add_argument("--use-ema", action="store_true", help="Only relevant if --checkpoint is given.")
    p.add_argument("--out-dir", type=Path, required=True, help="Also used as the directory to watch in --monitor mode.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32", "int8"], default="fp32",
                    help="Storage dtype on disk. Default is fp32 -- bit-exact match to what rae.encode() "
                         "actually produces at --compute-dtype fp32, i.e. zero extra quantization noise for "
                         "the decoder to deal with. fp16/int8 are opt-in space-savers (fp16: ~2x smaller, "
                         "int8: ~4x smaller) but introduce real reconstruction-quality risk -- if you go that "
                         "route, verify with verify_latents_recon.py --compare-online before committing to a "
                         "full run. bf16 is stored as fp32 in the memmap since numpy has no native bf16 dtype.")
    p.add_argument("--calib-samples", type=int, default=2048,
                    help="Number of images (across all ranks) used to calibrate per-channel int8 scales. "
                         "Ignored unless --dtype int8.")
    p.add_argument("--compute-dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="Precision used to RUN rae.encode(). Default fp32 matches train.py, where "
                         "`z = rae.encode(images)` runs outside any autocast context. Only change this if "
                         "you've also changed train.py's stage-2 encode call to use autocast, otherwise the "
                         "saved latents won't match what online encoding would have produced.")
    p.add_argument("--classes-per-group", type=int, default=None,
                    help="If set, only extract this many classes in this run (e.g. 250 out of 1000), selected "
                         "via --group-idx. Lets you extract in stages to cap peak disk usage instead of needing "
                         "space for the full dataset at once. Output for this run is written to "
                         "<out-dir>/group<NNN>/ so groups never collide with each other.")
    p.add_argument("--group-idx", type=int, default=None,
                    help="Which class-group to extract (0-indexed). Required if --classes-per-group is set. "
                         "E.g. with --classes-per-group 250 and 1000 total classes, valid values are 0,1,2,3.")
    p.add_argument("--monitor", action="store_true",
                    help="Don't extract anything -- just watch --out-dir for progress_rank*.json files written "
                         "by a (possibly separate, concurrently running) extraction process and print a "
                         "refreshing progress table. Doesn't need torchrun/GPUs: `python extract_latents.py "
                         "--monitor --out-dir <same out-dir>`. Run this in a second terminal/tmux pane.")
    p.add_argument("--refresh", type=float, default=5.0, help="Seconds between refreshes. Only used with --monitor.")
    p.add_argument("--tmux", action="store_true",
                    help="One-command launch: auto-creates a tmux session with extraction running in one pane "
                         "and --monitor running live in a split pane, then attaches you to it. No separate "
                         "script, no manually opening a second pane -- just: `python extract_latents.py --tmux "
                         "--nproc-per-node 8 --config ... --data-path ... --out-dir ...` (plain `python`, not "
                         "torchrun -- this process re-invokes torchrun itself inside the tmux pane). Requires "
                         "the `tmux` binary to be installed.")
    p.add_argument("--nproc-per-node", type=int, default=1,
                    help="GPUs to use. Only consumed by --tmux mode (normal torchrun invocations set this via "
                         "`torchrun --nproc_per_node=N`, not this flag).")
    args = p.parse_args()
    if not (args.monitor or args.tmux):
        if args.config is None or args.data_path is None:
            p.error("--config and --data-path are required unless --monitor or --tmux is set.")
    elif args.tmux and (args.config is None or args.data_path is None):
        p.error("--config and --data-path are required (they get passed through to the extraction process).")
    return args


TORCH_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
# numpy has no native bf16 storage type -> fall back to fp32 on disk for bf16 requests.
# int8 storage doesn't have a single numpy dtype here -- values are stored as np.int8,
# but need an accompanying per-channel scale (computed via calibration) to dequantize.
NUMPY_DTYPE = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "int8": np.int8}


def stage2_matched_transform(image_size: int) -> transforms.Compose:
    """
    Same preprocessing as train.py's `stage2_transform`, minus RandomHorizontalFlip
    (a random augmentation that can't be baked into a fixed latent cache). This is
    what actually needs to match -- NOT stage-1's training-time RandomCrop pipeline.
    """
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# --monitor mode: watch progress_rank*.json files written by a (possibly
# separate, concurrently running) extraction process. No GPU/torchrun needed.
# Run in a second terminal/tmux pane: `python extract_latents.py --monitor --out-dir <same out-dir>`
# ---------------------------------------------------------------------------

def _find_progress_files(out_dir: Path):
    return sorted(out_dir.rglob("progress_rank*.json"))


def _render_progress(out_dir: Path) -> None:
    files = _find_progress_files(out_dir)
    print(f"Monitoring: {out_dir}\n")
    if not files:
        print("No progress files found yet -- extraction may not have started writing, or --out-dir is wrong.")
        return

    now = time.time()
    header = f"{'shard':45s} {'progress':>20s} {'rate(img/s)':>12s} {'updated':>10s}"
    print(header)
    print("-" * len(header))

    total_done, total_target = 0, 0
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # file is being written concurrently -- just skip this refresh
        write_ptr = data.get("write_ptr", 0)
        local_n = data.get("local_n", 0)
        rate = data.get("rate_imgs_per_sec", 0.0)
        updated_at = data.get("updated_at", now)
        done = data.get("completed", False)
        age = now - updated_at
        rel = str(f.relative_to(out_dir))
        pct = (write_ptr / local_n * 100) if local_n else 0.0
        status = "DONE" if done else f"{age:.0f}s ago"
        print(f"{rel:45s} {write_ptr:>8d}/{local_n:<8d}({pct:5.1f}%) {rate:>12.1f} {status:>10s}")
        total_done += write_ptr
        total_target += local_n

    print("-" * len(header))
    if total_target > 0:
        overall_pct = total_done / total_target * 100
        print(f"OVERALL: {total_done}/{total_target} ({overall_pct:.1f}%)")


def run_monitor(out_dir: Path, refresh: float) -> None:
    try:
        while True:
            os.system("clear")
            _render_progress(out_dir)
            print(f"\n(refreshing every {refresh:.0f}s -- Ctrl+C to stop)")
            time.sleep(refresh)
    except KeyboardInterrupt:
        pass


def launch_in_tmux(args: argparse.Namespace) -> None:
    """
    One-command experience: build a tmux session with extraction in one pane
    and `--monitor` live in a split pane, then hand off to `tmux attach` via
    execvp (this process becomes tmux; nothing left running in the background
    unexpectedly). All still one file -- this just shells out to the `tmux`
    binary the same way a human would from the command line.
    """
    if shutil.which("tmux") is None:
        raise RuntimeError(
            "--tmux requires the `tmux` binary to be installed and on PATH. "
            "Install it (e.g. `apt install tmux`) or drop --tmux and run the extraction "
            "and `--monitor` yourself in two panes/terminals."
        )

    this_file = os.path.abspath(__file__)
    session = "rae_extract"

    # Rebuild the argv for the actual extraction process: everything the user passed,
    # minus --tmux and --nproc-per-node (torchrun owns process count, not this script).
    passthrough = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--tmux":
            continue
        if a == "--nproc-per-node":
            skip_next = True
            continue
        passthrough.append(a)

    extract_cmd = (
        f"torchrun --standalone --nproc_per_node={args.nproc_per_node} "
        f"{shlex.quote(this_file)} {' '.join(shlex.quote(a) for a in passthrough)}; "
        f"echo; echo '[extraction exited -- press Enter to close this pane]'; read"
    )
    monitor_cmd = (
        f"sleep 3; python {shlex.quote(this_file)} --monitor --out-dir {shlex.quote(str(args.out_dir))} --refresh 5"
    )

    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "extract", extract_cmd], check=True)
    subprocess.run(["tmux", "split-window", "-v", "-t", session, monitor_cmd], check=True)
    subprocess.run(["tmux", "select-pane", "-t", f"{session}.0"], check=True)
    os.execvp("tmux", ["tmux", "attach", "-t", session])  # replaces this process -- clean handoff


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.monitor:
        run_monitor(args.out_dir, args.refresh)
        return

    if args.tmux:
        launch_in_tmux(args)
        return  # unreachable in practice (execvp replaces the process), kept for clarity

    rank, world_size, device = setup_distributed()

    full_cfg = OmegaConf.load(args.config)
    (rae_config, *_) = parse_configs(full_cfg)

    # ---- model init: identical to train_stage1.py's encoder setup ----
    # The encoder is frozen for the entirety of stage-1 training (see train_stage1.py:
    # rae.encoder.requires_grad_(False), never touched by optimizer.step()), so its
    # weights come entirely from instantiate_from_config(rae_config) -- no checkpoint
    # needed. --checkpoint is only for the unusual case where pretrained encoder
    # weights aren't baked into the config itself.
    rae: RAE = instantiate_from_config(rae_config).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["ema"] if (args.use_ema and "ema" in ckpt) else ckpt["model"]
        missing, unexpected = rae.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"Loaded checkpoint (use_ema={args.use_ema}). missing={len(missing)}, unexpected={len(unexpected)}")
    elif rank == 0:
        print("No --checkpoint given: using encoder weights as instantiated directly from --config.")
    rae.eval()
    rae.requires_grad_(False)

    if rank == 0:
        # Diagnostic only: doesn't prove the weights are the *correct* pretrained
        # weights, but lets you sanity-check param count against the expected
        # architecture (DINOv2-B encoder ~86M params) and compare a checksum across
        # runs / against a reference load of the same pretrained checkpoint elsewhere.
        enc_params = sum(p.numel() for p in rae.encoder.parameters())
        enc_checksum = sum(p.detach().abs().sum().item() for p in rae.encoder.parameters())
        print(f"[diagnostic] encoder param count: {enc_params/1e6:.2f}M, "
              f"checksum(sum |w|): {enc_checksum:.4f}")
        print(f"[diagnostic] rae_config encoder section:\n{OmegaConf.to_yaml(rae_config)}")

    compute_dtype = TORCH_DTYPE[args.compute_dtype]
    compute_autocast_enabled = args.compute_dtype != "fp32"
    np_dtype = NUMPY_DTYPE[args.dtype]

    dataset_full = ImageFolder(str(args.data_path), transform=stage2_matched_transform(args.image_size))
    class_names = dataset_full.classes  # always the full 1000-class list, regardless of grouping

    subset_to_full_idx = None  # None => `dataset` indices already are full-ImageFolder indices
    if args.classes_per_group is not None:
        if args.group_idx is None:
            raise ValueError("--group-idx is required when --classes-per-group is set.")
        total_classes = len(class_names)
        start_cls = args.group_idx * args.classes_per_group
        end_cls = min(start_cls + args.classes_per_group, total_classes)
        if start_cls >= total_classes:
            raise ValueError(
                f"--group-idx {args.group_idx} is out of range: {total_classes} classes / "
                f"{args.classes_per_group} per group only gives groups 0..{(total_classes - 1) // args.classes_per_group}."
            )
        keep_indices = [i for i, (_, target) in enumerate(dataset_full.samples) if start_cls <= target < end_cls]
        dataset = Subset(dataset_full, keep_indices)
        subset_to_full_idx = np.array(keep_indices, dtype=np.int64)
        if rank == 0:
            print(f"[class-group] group {args.group_idx}: classes [{start_cls}, {end_cls}) of {total_classes} "
                  f"-> {len(dataset)} images")
    else:
        dataset = dataset_full

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
    )

    local_indices = list(sampler)
    local_n = len(local_indices)

    # ---- probe output latent shape once, so we can preallocate the memmap ----
    probe_img = dataset[local_indices[0]][0].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=compute_autocast_enabled):
        probe_latent = rae.encode(probe_img)
    latent_shape = tuple(probe_latent.shape[1:])
    bytes_per_sample = int(np.prod(latent_shape)) * np.dtype(np_dtype).itemsize
    total_bytes = bytes_per_sample * local_n
    if rank == 0:
        est_total = bytes_per_sample * len(dataset)
        scope_label = f"this group ({len(dataset)} images)" if args.classes_per_group is not None else "full dataset"
        print(f"Latent shape per sample: {latent_shape}, dtype on disk: {np_dtype}")
        print(f"Per-rank shard size: ~{total_bytes / 1e9:.2f} GB | Estimated size for {scope_label}: ~{est_total / 1e9:.2f} GB")

    effective_out_dir = args.out_dir / f"group{args.group_idx:03d}" if args.classes_per_group is not None else args.out_dir
    effective_out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # make sure the directory exists on shared storage before any rank writes into it

    # ---- calibration pass for int8: compute per-channel symmetric scale ----
    # scale[c] = max_abs_over_calib_set(latent[:, c, ...]) / 127
    # Every rank calibrates on its own slice of --calib-samples images, then an
    # all_reduce(MAX) synchronizes so every rank (and every shard) uses the exact
    # same scale -- required since shards are read back as one logical dataset.
    channel_scale = None
    if args.dtype == "int8":
        n_channels = latent_shape[0]
        local_calib_n = min(args.calib_samples // max(world_size, 1), local_n)
        local_calib_n = max(local_calib_n, 1)
        running_max_abs = torch.zeros(n_channels, device=device)
        calib_seen = 0
        for images, _ in loader:
            if calib_seen >= local_calib_n:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=compute_autocast_enabled):
                latents = rae.encode(images).float()
            batch_max = latents.abs().amax(dim=tuple(range(2, latents.dim())))  # (B, C)
            running_max_abs = torch.maximum(running_max_abs, batch_max.amax(dim=0))
            calib_seen += latents.shape[0]
        dist.all_reduce(running_max_abs, op=dist.ReduceOp.MAX)
        channel_scale = (running_max_abs / 127.0).clamp(min=1e-8).cpu().numpy().astype(np.float32)
        if rank == 0:
            print(f"[int8 calib] per-channel scale computed over ~{args.calib_samples} images. "
                  f"scale range: [{channel_scale.min():.6f}, {channel_scale.max():.6f}]")
            np.save(effective_out_dir / "channel_scale.npy", channel_scale)
        # broadcast the array object itself so every rank has an identical copy in memory
        dist.barrier()

    shard_path = effective_out_dir / f"latents_rank{rank:03d}.dat"
    label_path = effective_out_dir / f"labels_rank{rank:03d}.npy"
    index_path = effective_out_dir / f"global_index_rank{rank:03d}.npy"

    # w+ preallocates the full file on disk up front (sequential writes below,
    # no fragmentation, no incremental resize).
    mm = np.memmap(shard_path, dtype=np_dtype, mode="w+", shape=(local_n, *latent_shape))
    labels = np.empty((local_n,), dtype=np.int64)
    # local_indices are indices into `dataset` (subset-local if class-grouped, else already
    # full-ImageFolder indices). Map through subset_to_full_idx so global_idx always means
    # "index into the full ImageFolder", matching what verify_latents_recon.py /
    # recover_partial_extraction.py expect when they rebuild the full ImageFolder to cross-check.
    if subset_to_full_idx is not None:
        global_idx = subset_to_full_idx[np.array(local_indices, dtype=np.int64)]
    else:
        global_idx = np.array(local_indices, dtype=np.int64)

    write_ptr = 0
    t0 = time.time()
    channel_scale_t = torch.from_numpy(channel_scale).to(device).view(1, -1, 1, 1) if channel_scale is not None else None
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=compute_autocast_enabled):
            latents = rae.encode(images)
        latents = latents.to(torch.float32)
        if args.dtype == "int8":
            quantized = torch.round(latents / channel_scale_t).clamp(-127, 127)
            latents_np = quantized.cpu().numpy().astype(np.int8)
        else:
            latents_np = latents.cpu().numpy().astype(np_dtype)
        bsz = latents_np.shape[0]
        mm[write_ptr:write_ptr + bsz] = latents_np
        labels[write_ptr:write_ptr + bsz] = targets.numpy()
        write_ptr += bsz
        if (write_ptr // args.batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = write_ptr / max(elapsed, 1e-6)
            # progress file for every rank (for monitor_extraction.py, e.g. run in a separate tmux pane)
            progress_path = effective_out_dir / f"progress_rank{rank:03d}.json"
            with open(progress_path, "w") as f:
                json.dump({
                    "write_ptr": write_ptr,
                    "local_n": local_n,
                    "rate_imgs_per_sec": rate,
                    "updated_at": time.time(),
                }, f)
            if rank == 0:
                print(f"[rank0] {write_ptr}/{local_n} ({rate:.1f} img/s)")

    elapsed = time.time() - t0
    with open(effective_out_dir / f"progress_rank{rank:03d}.json", "w") as f:
        json.dump({
            "write_ptr": write_ptr,
            "local_n": local_n,
            "rate_imgs_per_sec": write_ptr / max(elapsed, 1e-6),
            "updated_at": time.time(),
            "completed": True,
        }, f)

    mm.flush()
    del mm  # release the memmap so the OS finalizes the file
    np.save(label_path, labels)
    np.save(index_path, global_idx)

    if rank == 0:
        meta = {
            "latent_shape": list(latent_shape),
            "dtype": args.dtype,
            "compute_dtype": args.compute_dtype,
            "world_size": world_size,
            "total_samples": len(dataset),
            "classes": class_names,
            "class_group": {
                "group_idx": args.group_idx,
                "classes_per_group": args.classes_per_group,
                "class_range": [start_cls, end_cls],
            } if args.classes_per_group is not None else None,
            "shard_filename_pattern": "latents_rank{:03d}.dat",
            "label_filename_pattern": "labels_rank{:03d}.npy",
            "index_filename_pattern": "global_index_rank{:03d}.npy",
            "quantization": {
                "scheme": "per_channel_symmetric_int8",
                "scale_file": "channel_scale.npy",
                "dequant_formula": "value_fp32 = int8_value.astype(float32) * channel_scale[channel]",
            } if args.dtype == "int8" else None,
        }
        with open(effective_out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta.json to {effective_out_dir}")

    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()