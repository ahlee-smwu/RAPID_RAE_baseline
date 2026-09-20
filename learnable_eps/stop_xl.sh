#!/usr/bin/env bash
# Stop a running learnable_eps/train.py launch (all ranks + torchrun + run_xl.sh).
# The bracket trick keeps pkill from matching this script's own command line.
pkill -f "^(bash |/bin/bash |/usr/bin/bash )?learnable_eps/run_xl.sh" 2>/dev/null   # anchored: only the launcher itself, not wrappers that mention it
pkill -f "[t]rain.py --config" 2>/dev/null
sleep 5
if pgrep -f "[t]rain.py --config" >/dev/null; then
  pkill -9 -f "[t]rain.py --config"; pkill -9 -f "[t]orchrun --standalone"
  sleep 3
fi
echo "remaining train.py processes: $(pgrep -fc '[t]rain.py --config')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
