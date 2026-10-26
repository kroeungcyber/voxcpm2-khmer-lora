# VoxCPM2 Khmer LoRA fine-tuning image.
# Base image matches the official VoxCPM training Dockerfile:
#   pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/models \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone + install VoxCPM (provides the `voxcpm` package AND the training script).
WORKDIR /workspace
RUN git clone https://github.com/OpenBMB/VoxCPM.git /workspace/VoxCPM
WORKDIR /workspace/VoxCPM
RUN pip install --no-cache-dir -e .

# Project files (config, scripts, data prep, prompts).
WORKDIR /workspace
COPY . /workspace/

RUN chmod +x /workspace/scripts/train.sh /workspace/runpod/start.sh

# Entrypoint: sanity-check GPU + data, then train.
CMD ["/workspace/runpod/start.sh"]
