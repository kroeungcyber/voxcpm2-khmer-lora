#!/usr/bin/env bash
set -euo pipefail
cd /workspace

echo "==> GPU check"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "No CUDA GPU detected. VoxCPM2 LoRA training requires an NVIDIA GPU (16GB+ VRAM)."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==> Data check"
if [ ! -f /workspace/data/train.jsonl ]; then
  echo "!! /workspace/data/train.jsonl not found."
  echo "   Build it first — see data/README.md (run data/make_manifest.py)."
  exit 1
fi

echo "==> Starting LoRA training"
exec bash /workspace/scripts/train.sh
