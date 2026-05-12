#!/usr/bin/env bash
set -euo pipefail

# entrypoint for TRAIN docker
# Required env vars:
#   STAGE = stage1 | stage2 | stage3_l3
#
# Mounts (provided by docker-compose):
#   /workspace             — repo (read/write)
#   /mnt/data              — training data
#   /mnt/models            — model cache
#   /mnt/output            — training outputs

STAGE="${STAGE:-stage1}"
CONFIG_DIR="/workspace/configs/tombench-rlvr"
CONFIG_NAME="rlvr_config_${STAGE}"

echo "=========================================="
echo "TRAIN entrypoint"
echo "  stage:  ${STAGE}"
echo "  config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "  CUDA visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo "  GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================="

# Verify data + models present
test -f "/mnt/data/tom_train.jsonl" || { echo "ERROR: /mnt/data/tom_train.jsonl missing"; exit 1; }
test -f "/mnt/data/tombench_eval_subset500.jsonl" || { echo "ERROR: subset500 missing"; exit 1; }

# Install ROLL in editable mode (idempotent)
pip install -e /workspace/framework/ROLL >/dev/null

# Run training
cd /workspace/framework/ROLL
exec python examples/start_rlvr_pipeline.py \
  --config_path "${CONFIG_DIR}" \
  --config_name "${CONFIG_NAME}"
