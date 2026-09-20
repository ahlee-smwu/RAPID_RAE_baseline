#!/usr/bin/env bash
# Launch experiment 0 (baseline) or 1 (rapid) on 3 GPUs, XL config, fp32 (the RAE authors recommend fp32 over bf16; a bf16 run diverged at epoch 9).
#   bash learnable_eps/run_xl.sh baseline|rapid [extra train.py args...]
# GPUs: $GPUS are CUDA ids (default 0,1,3), $NPROC (default 3) must equal the number listed; $PRECISION defaults to fp32.
# CUDA orders devices fastest-first on this host: CUDA 0,1,2,3,4 = nvidia-smi 2,0,1,3,4 (the Ada is CUDA 0 / smi 2).
# So GPUS=0,3,4 runs on nvidia-smi 2,3,4 and GPUS=1,2 on nvidia-smi 0,1. Verify with nvidia-smi --query-compute-apps.
set -euo pipefail
cd "$(dirname "$0")/.."
arm=${1:?baseline|rapid}; shift || true
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,3}
export LAT=${LAT:-/home/ahlee/latents_fp16}   # fp16 copy (convert_latents_fp16.py); fp32 original: /mnt/aisha/ahlee-rae
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs_rapid
exec torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-3} \
  learnable_eps/train.py \
  --config learnable_eps/configs/DiTDH-XL_DINOv2-B_${arm}.yaml \
  --latent-path "$LAT" \
  --results-dir ckpts/stage2 \
  --precision ${PRECISION:-fp32} --compile --global-seed 42 "$@"
