# DEV image — used on macOS for data construction, synthesis, eval, analysis.
# No CUDA. Lean Python 3.10 base.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/dev/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Mount points (mounted at runtime via compose)
VOLUME ["/workspace/scripts", "/workspace/data", "/workspace/output", "/workspace/configs"]

CMD ["bash"]
