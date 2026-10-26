#!/usr/bin/env bash
set -euo pipefail

# Personalized Khmer voice — LoRA fine-tuning of VoxCPM2.
# Run on a RunPod GPU pod with the project at /workspace.

WORKSPACE="${WORKSPACE:-/workspace}"
VOXCPM_REPO="${WORKSPACE}/VoxCPM"
MODEL_DIR="${WORKSPACE}/models/VoxCPM2"
CONFIG="${WORKSPACE}/configs/finetune_lora_khmer.yaml"

echo "==> [1/4] Cloning OpenBMB/VoxCPM"
if [ ! -d "${VOXCPM_REPO}/.git" ]; then
  git clone https://github.com/OpenBMB/VoxCPM.git "${VOXCPM_REPO}"
else
  (cd "${VOXCPM_REPO}" && git pull --ff-only)
fi

echo "==> [2/4] Installing voxcpm (editable)"
cd "${VOXCPM_REPO}"
pip install --no-cache-dir -e .

echo "==> [3/4] Downloading VoxCPM2 weights (if missing)"
mkdir -p "${MODEL_DIR}"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM2", local_dir="/workspace/models/VoxCPM2")
PY
fi

echo "==> [4/4] Launching LoRA training"
# Relocate /workspace paths if WORKSPACE is overridden (local / non-RunPod runs).
RUNTIME_CONFIG="${WORKSPACE}/configs/runtime_lora.yaml"
sed -e "s#/workspace#${WORKSPACE}#g" "${CONFIG}" > "${RUNTIME_CONFIG}"
cd "${VOXCPM_REPO}"
exec python scripts/train_voxcpm_finetune.py --config_path "${RUNTIME_CONFIG}"
