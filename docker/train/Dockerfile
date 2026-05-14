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

# Install ROLL framework requirements.
# requirements_torch280_vllm.txt chains via `-r` to requirements_common.txt,
# which further chains to requirements_vision.txt and has a `./mcore_adapter`
# local-path entry. Copy all the requirements files and the mcore_adapter
# package into a working directory and pip-install from there so every
# relative reference resolves. The runtime ROLL source still comes via
# volume mount.
RUN mkdir -p /tmp/roll-reqs
COPY framework/ROLL/requirements_common.txt        /tmp/roll-reqs/
COPY framework/ROLL/requirements_torch280_vllm.txt /tmp/roll-reqs/
COPY framework/ROLL/requirements_vision.txt        /tmp/roll-reqs/
COPY framework/ROLL/mcore_adapter                  /tmp/roll-reqs/mcore_adapter
RUN cd /tmp/roll-reqs && pip install -r requirements_torch280_vllm.txt

# Pin click<8.3 because click 8.3.0 + Python 3.10 + ray triggers a deepcopy
# ValueError on `ray start --head`:
#   ValueError: <object object at 0x...> is not a valid Sentinel
# (See https://discuss.ray.io/t/ray-start-head-failing-due-to-not-valid-sentinel/23121)
# Some transitive dep of ROLL pulls in click 8.3, so we force a downgrade
# after the requirements install.
RUN pip install --no-deps 'click==8.2.1'

# Install ROLL in editable mode (source is mounted at runtime)
# We DON'T `pip install -e .` here because the source is mounted; instead
# we set PYTHONPATH at runtime via entrypoint.

ENV PYTHONPATH=/workspace:/workspace/framework/ROLL

# vLLM, megatron, flash-attn are typically already in the base image or in req.txt
# Verify imports during build:
RUN python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)" || \
    echo "vllm not yet importable; will resolve at runtime"

# Install entrypoint to /usr/local/bin so the volume mount at /workspace
# (which is the bind-mount of the host repo) doesn't shadow it. The host
# repo has docker/train/entrypoint.sh, so we could also reference that
# path at runtime, but a baked-in /usr/local/bin copy is more robust to
# layout changes.
COPY docker/train/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
