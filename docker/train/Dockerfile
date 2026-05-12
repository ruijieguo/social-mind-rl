# TRAIN image — runs on remote GPU server (16×H800)
# Based on NVIDIA's PyTorch container which already has CUDA + PyTorch matched.
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install ROLL framework requirements
COPY framework/ROLL/requirements_torch280_vllm.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

# Install ROLL in editable mode (source is mounted at runtime)
# We DON'T `pip install -e .` here because the source is mounted; instead
# we set PYTHONPATH at runtime via entrypoint.

ENV PYTHONPATH=/workspace:/workspace/framework/ROLL

# vLLM, megatron, flash-attn are typically already in the base image or in req.txt
# Verify imports during build:
RUN python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)" || \
    echo "vllm not yet importable; will resolve at runtime"

COPY docker/train/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
