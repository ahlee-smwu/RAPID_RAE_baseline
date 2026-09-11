# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
verify_latents_recon.py

Sanity-check the latents cached by extract_latents.py.

Two things this script does:
  1. Load latents directly from the .dat shards (via ShardedLatentDataset) and
     run them through rae.decode() -- exactly like the original stage-1
     reconstruction script, except skipping rae.encode() entirely since the
     latent is already sitting on disk.
  2. (Optional, --compare-online) For the same images, ALSO run a fresh
     rae.encode(image) right now and compare it directly against the cached
     latent (mean abs diff, cosine similarity) plus decode both and save a
     side-by-side grid so you can eyeball original / cached-latent-recon /
     online-recon together.

This is the real verification: matching PNGs alone only tells you the decoder
still works, not that the cached latent equals what stage-2 training would
actually see. --compare-online tells you that directly.

Usage (single or multi-GPU):
    torchrun --standalone --nproc_per_node=N verify_latents_recon.py \
        --config configs/stage1/training/DINOv2-B_decXL.yaml \
        --latent-dir /fast_local_disk/imagenet_latents \
        --sample-dir samples/latent_verify \
        --num-samples 64 \
        --per-proc-batch-size 8 \
        --precision fp32 \
        --compare-online --data-path <imagenet_train_split>

IMPORTANT: --precision here must match whatever --compute-dtype you used in
extract_latents.py (default was fp32/no-autocast, matching train.py's actual
online encode call). If they don't match, --compare-online will report a
nonzero diff that's just a precision mismatch, not a bug in the cached data.
"""

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(os.path.join(root_dir, 'stage1'))

import argparse
import math
import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np

from sample_ddp import create_npz_from_sample_folder
from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs, center_crop_arr  # same one used by extract_latents.py
from latent_dataset import ShardedLatentDataset


class IndexedLatentDataset(Dataset):
    """Wraps ShardedLatentDataset to also return the dataset-order index, so we
    can (a) name output files consistently and (b) look up the matching
    original image for --compare-online."""

    def __init__(self, latent_dataset: ShardedLatentDataset):
        self.latent_dataset = latent_dataset

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, index):
        latent, label = self.latent_dataset[index]
        return latent, label, index


def sanitize_component(component: str) -> str:
    return component.replace(os.sep, "-")


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()  # matches extract_latents.py -- disables the noising path in RAE.encode()

    latent_dataset = ShardedLatentDataset(args.latent_dir)
    if latent_dataset.meta.get("quantization"):
        if rank == 0:
            print("Cached latents are int8-quantized; ShardedLatentDataset dequantizes "
                  "them transparently, so nothing else changes here.")
    dataset = IndexedLatentDataset(latent_dataset)

    image_dataset: Optional[ImageFolder] = None
    if args.compare_online:
        if not args.data_path:
            raise ValueError("--compare-online requires --data-path (the same ImageFolder root used for extraction).")
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ])
        image_dataset = ImageFolder(args.data_path, transform=transform)

    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No latents found at {args.latent_dir}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    selected_indices = list(range(requested))
    rank_indices = selected_indices[rank::world_size]
    subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    folder_components: List[str] = ["latent-verify", f"bs{args.per_proc_batch_size}", args.precision]
    folder_name = "-".join(folder_components)
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving reconstructed samples at {sample_folder_dir}")
    dist.barrier()

    loader = DataLoader(
        subset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    local_total = len(rank_indices)
    iterator = tqdm(loader, desc="Latent recon", total=math.ceil(local_total / args.per_proc_batch_size)) if rank == 0 else loader

    # running comparison stats (only used if --compare-online)
    sum_abs_diff = torch.zeros(1, device=device)
    sum_sq_diff = torch.zeros(1, device=device)
    sum_cos_sim = torch.zeros(1, device=device)
    n_compared = torch.zeros(1, device=device)
    saved_comparison_grid = False

    with torch.inference_mode():
        for latents, labels, idxs in iterator:
            if latents.numel() == 0:
                continue
            latents = latents.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                recon = rae.decode(latents)
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            idxs_list = idxs.tolist()
            for sample, idx in zip(recon_np, idxs_list):
                Image.fromarray(sample).save(f"{sample_folder_dir}/{idx:06d}.png")

            if args.compare_online:
                orig_indices = [latent_dataset.original_imagefolder_index(i) for i in idxs_list]
                images = torch.stack([image_dataset[oi][0] for oi in orig_indices]).to(device, non_blocking=True)
                with autocast(**autocast_kwargs):
                    online_latents = rae.encode(images)
                    online_recon = rae.decode(online_latents).clamp(0, 1)

                cached = latents.float()
                online = online_latents.float()
                diff = (cached - online).flatten(1)
                sum_abs_diff += diff.abs().sum()
                sum_sq_diff += (diff ** 2).sum()
                sum_cos_sim += F.cosine_similarity(cached.flatten(1), online.flatten(1), dim=1).sum()
                n_compared += cached.shape[0]

                if rank == 0 and not saved_comparison_grid:
                    k = min(4, images.shape[0])
                    grid = make_grid(
                        torch.cat([images[:k].cpu(), recon[:k].cpu(), online_recon[:k].cpu()], dim=0),
                        nrow=k,
                    )
                    grid_img = (grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy())
                    Image.fromarray(grid_img).save(os.path.join(sample_folder_dir, "_comparison_grid_original-cached-online.png"))
                    saved_comparison_grid = True

    dist.barrier()

    if args.compare_online:
        dist.all_reduce(sum_abs_diff)
        dist.all_reduce(sum_sq_diff)
        dist.all_reduce(sum_cos_sim)
        dist.all_reduce(n_compared)
        if rank == 0:
            n = n_compared.item()
            mean_abs_diff = (sum_abs_diff / (n * latent_dataset.latent_shape[0] * latent_dataset.latent_shape[1] * latent_dataset.latent_shape[2])).item()
            rmse = math.sqrt((sum_sq_diff / (n * latent_dataset.latent_shape[0] * latent_dataset.latent_shape[1] * latent_dataset.latent_shape[2])).item())
            mean_cos_sim = (sum_cos_sim / n).item()
            print("\n=== cached latent vs. fresh online rae.encode() ===")
            print(f"  compared samples : {int(n)}")
            print(f"  mean |diff|      : {mean_abs_diff:.6f}")
            print(f"  RMSE             : {rmse:.6f}")
            print(f"  mean cosine sim  : {mean_cos_sim:.6f}  (should be ~1.0)")
            if mean_cos_sim < 0.999:
                print("  WARNING: cosine similarity is noticeably below 1.0 -- check that "
                      "--precision here matches --compute-dtype used during extraction, and "
                      "that the image preprocessing matches (center_crop_arr, no random flip).")
            else:
                print("  Looks good: cached latents match fresh online encoding closely.")

    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, requested)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the stage-1 config file.")
    parser.add_argument("--latent-dir", type=str, required=True, help="Directory containing the cached .dat shards (extract_latents.py --out-dir).")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store reconstructed samples.")
    parser.add_argument("--per-proc-batch-size", type=int, default=8, help="Number of latents processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of cached samples to reconstruct (defaults to 64 for a quick check; pass a larger number or omit for a fuller run).")
    parser.add_argument("--image-size", type=int, default=256, help="Only used for --compare-online preprocessing; must match extraction's --image-size.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32",
                         help="Autocast precision for decode (and encode, if --compare-online). Should match "
                              "--compute-dtype used in extract_latents.py for a meaningful --compare-online result.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-online", action="store_true",
                         help="Also re-encode the original images right now and compare against the cached latent.")
    parser.add_argument("--data-path", type=str, default=None, help="Required if --compare-online: same ImageFolder root used for extraction.")
    args = parser.parse_args()
    main(args)