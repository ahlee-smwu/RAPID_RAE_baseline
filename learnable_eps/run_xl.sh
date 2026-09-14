#!/usr/bin/env bash
# Launch experiment 0 (baseline) or 1 (rapid) on 2 GPUs, XL config.
#   bash learnable_eps/run_xl.sh baseline|rapid [extra train.py args...]
# GPUs: $GPUS (default 0,1), $NPROC (default 2) must equal the number of GPUs listed.
set -euo pipefail
cd "$(dirname "$0")/.."
arm=${1:?baseline|rapid}; shift || true
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
export LAT=${LAT:-/home/ahlee/latents_fp16}   # fp16 copy (convert_latents_fp16.py); fp32 original: /mnt/aisha/ahlee-rae
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs_rapid
exec torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-2} \
  learnable_eps/train.py \
  --config learnable_eps/configs/DiTDH-XL_DINOv2-B_${arm}.yaml \
  --latent-path "$LAT" \
  --results-dir ckpts/stage2 \
  --precision bf16 --compile --global-seed 42 "$@"
