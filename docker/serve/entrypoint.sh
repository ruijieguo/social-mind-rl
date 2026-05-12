#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/output/final_model}"
PORT="${SERVE_PORT:-8000}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen3-8b-tom}"

echo "=========================================="
echo "SERVE entrypoint"
echo "  model: ${MODEL_PATH}"
echo "  port:  ${PORT}"
echo "  name:  ${SERVED_NAME}"
echo "=========================================="

test -d "${MODEL_PATH}" || { echo "ERROR: model dir ${MODEL_PATH} missing"; exit 1; }

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --served-model-name "${SERVED_NAME}"
