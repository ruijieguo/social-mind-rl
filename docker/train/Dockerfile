# TRAIN image — runs on remote GPU server.
# Mirrors ROLL's official framework/ROLL/docker/Dockerfile.torch260.vllm
# install ordering (NVIDIA pytorch:24.05 base + explicit version pins),
# because the requirements_*.txt files have unresolvable transitive conflicts
# (vllm 0.8.4 vs ray[default] 2.48 on opentelemetry-sdk).
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

# Remove preinstalled torch/TE/flash-attn so we can install matched versions
# below. cudf/dask packages are removed because they pin numpy too tightly.
RUN pip uninstall -y torch torchvision torch-tensorrt \
    flash_attn transformer-engine \
    cudf dask-cuda cugraph cugraph-service-server cuml raft-dask \
    cugraph-dgl cugraph-pyg dask-cudf || true

# Pin torch 2.6 + matching torchvision/audio (CUDA 12.4 wheels).
# download.pytorch.org from CN is flaky (200+ MB wheel often disconnects
# mid-stream), so prefetch via Aliyun's pytorch-wheels mirror with curl
# --continue-at to support resumption, then pip install from the local files.
RUN mkdir -p /tmp/torch-wheels && cd /tmp/torch-wheels && \
    for f in \
      torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl \
      torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl \
      torchaudio-2.6.0+cu124-cp310-cp310-linux_x86_64.whl ; do \
        echo "downloading $f ..." && \
        curl -fL --retry 20 --retry-delay 5 --retry-all-errors -C - \
             -o "$f" "https://mirrors.aliyun.com/pytorch-wheels/cu124/$f" ; \
    done && \
    pip install ./torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl \
                ./torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl \
                ./torchaudio-2.6.0+cu124-cp310-cp310-linux_x86_64.whl && \
    rm -rf /tmp/torch-wheels

# Pin file used by ALL subsequent `pip install` calls so transitive deps
# (especially deepspeed / vllm / megatron) cannot upgrade torch. Without
# this, pip's resolver picks torch 2.12+cu130 from the latest sdist and
# completely undoes the cu124 install above.
RUN echo "torch==2.6.0+cu124" >  /tmp/constraints.txt && \
    echo "torchvision==0.21.0+cu124" >> /tmp/constraints.txt && \
    echo "torchaudio==2.6.0+cu124" >> /tmp/constraints.txt && \
    echo "transformer-engine==2.2.0" >> /tmp/constraints.txt && \
    echo "transformer-engine-torch==2.2.0" >> /tmp/constraints.txt && \
    echo "transformer-engine-cu12==2.2.0" >> /tmp/constraints.txt && \
    echo "deepspeed==0.16.4" >> /tmp/constraints.txt && \
    echo "vllm==0.8.4" >> /tmp/constraints.txt && \
    echo "megatron-core==0.16.0" >> /tmp/constraints.txt && \
    echo "transformers<5.0" >> /tmp/constraints.txt && \
    echo "transformers>=4.51,<4.55" >> /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# Replace preinstalled opencv with headless build to avoid GL/X11 deps.
RUN pip uninstall -y opencv opencv-python opencv-python-headless || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ && \
    pip install opencv-python-headless==4.11.0.86

# Core ML deps with versions matched to ROLL's validated combination.
# - --no-build-isolation so TE/deepspeed setup.py sees the installed torch
# - --no-deps because pip's dep resolver keeps upgrading torch to the
#   latest cu130 build (2.12.0) via transitive deps from deepspeed/TE.
# Install the runtime deps these packages actually need ourselves first.
RUN pip install --no-build-isolation \
      "numpy==1.26.4" "optree>=0.13.0" "spacy==3.7.5" "weasel==0.4.1" \
      "psutil" "py-cpuinfo" "hjson" "ninja" "pydantic" \
      "packaging" "pyyaml" "msgpack" "tqdm"
RUN pip install --no-build-isolation --no-deps \
      transformer-engine==2.2.0 \
      transformer-engine-torch==2.2.0 \
      transformer-engine-cu12==2.2.0
RUN pip install --no-build-isolation --no-deps \
      megatron-core==0.16.0 \
      deepspeed==0.16.4

# Flash-attn 2.7.x — DISABLED. Building from sdist consistently deadlocks
# at file [12/22] of the 85-file CUDA compile (no error, no progress, no
# nvcc subprocess). Both MAX_JOBS=16 and MAX_JOBS=4 deadlock identically.
# Prebuilt wheels live only on GitHub Releases which is unreachable from
# this CN datacenter; every proxy mirror returns reset-by-peer.
#
# vLLM 0.8.4 bundles its own flash-attention kernels (vllm-flash-attn),
# so the pip flash-attn package is not strictly required for inference.
# ROLL falls back to torch SDPA for paths that cannot use flash-attn.
# We accept the modest perf hit to unblock the pipeline.
# RUN MAX_JOBS=4 NVCC_THREADS=1 \
#     pip install --no-build-isolation --no-cache-dir --verbose \
#       flash-attn==2.7.2.post1

# vLLM separately (it pins ray + many transitive deps).
RUN pip install vllm==0.8.4

# NVIDIA Apex (fused optimizers) — disabled because git clone of apex
# submodules (cudnn-frontend, cutlass) is unreliable from this network
# ("RPC failed; curl 56 GnuTLS recv error" / "early EOF"). Megatron-core
# falls back to non-fused optimizers automatically when apex is absent;
# the perf hit on a single 8-GPU node is small relative to the install
# friction. Re-enable once a closer mirror is identified.
# ARG APEX_URL=git+https://github.com/NVIDIA/apex.git@25.04
# RUN pip uninstall -y apex || true && \
#     MAX_JOBS=32 NINJA_FLAGS="-j32" NVCC_APPEND_FLAGS="--threads 32" \
#     pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
#       --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 32" \
#       ${APEX_URL}

# Now install the ROLL python-only deps from requirements_common.txt, which
# pull in ray, dacite, hydra, datasets, etc. We skip the `./mcore_adapter`
# editable entry to avoid bind-mount writes; instead we install mcore_adapter
# explicitly below from a COPY'd snapshot.
COPY framework/ROLL/requirements_common.txt /tmp/roll-reqs/requirements_common.txt
COPY framework/ROLL/requirements_vision.txt /tmp/roll-reqs/requirements_vision.txt
COPY framework/ROLL/mcore_adapter          /tmp/roll-reqs/mcore_adapter
RUN sed -i '/^\.\/mcore_adapter$/d' /tmp/roll-reqs/requirements_common.txt && \
    cd /tmp/roll-reqs && pip install -r requirements_common.txt
RUN pip install /tmp/roll-reqs/mcore_adapter --no-deps

# Pin click<8.3 because click 8.3.0 + Python 3.10 + ray triggers a deepcopy
# ValueError on `ray start --head`. See:
# https://discuss.ray.io/t/ray-start-head-failing-due-to-not-valid-sentinel/23121
RUN pip install --no-deps 'click==8.2.1'

# Pin setuptools<75 so pkg_resources is still bundled with setuptools.
# vllm 0.8.4's TP>1 ray-distributed-executor path uses pkg_resources at
# runtime (see vllm/executor/ray_distributed_executor.py:_check_ray_cgraph_installation).
# setuptools>=75 split pkg_resources into its own distribution, breaking
# that import. Symptoms: ModuleNotFoundError pkg_resources at the moment
# vllm tries to compile a Ray DAG (i.e. the first inference batch with TP>1).
# Affects 14B training but not 8B (8B uses TP=1, no Ray DAG path).
RUN pip install --no-deps 'setuptools<75'

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
