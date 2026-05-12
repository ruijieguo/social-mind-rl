#!/usr/bin/env bash
set -euo pipefail

source configs/deploy.env

echo "[remote-env-check] checking ${TRAIN_HOST} ..."
ssh -i "${TRAIN_SSH_KEY}" "${TRAIN_HOST}" bash -c "'
echo \"--- nvidia-smi ---\"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo \"--- docker ---\"
docker --version
docker compose version
echo \"--- disk ---\"
df -h \"${TRAIN_PATH}\" \"${TRAIN_OUTPUT_DIR}\" \"${TRAIN_DATA_DIR}\" \"${TRAIN_MODELS_DIR}\" 2>/dev/null || true
echo \"--- mounts ---\"
ls -la \"${TRAIN_PATH}\" 2>/dev/null | head -10 || echo \"(${TRAIN_PATH} not yet created)\"
'"
