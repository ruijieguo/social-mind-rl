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
case "${STAGE}" in
  stage1|stage1_1x8|stage1_2x8)
    DATA_FILE="/mnt/data/tom_train_4k.jsonl"
    ;;
  stage2|stage2_1x8|stage2_2x8|stage3_l3)
    DATA_FILE="/mnt/data/tom_train.jsonl"
    ;;
  *)
    echo "ERROR: unknown STAGE=${STAGE}"
    exit 1
    ;;
esac
test -f "${DATA_FILE}" || { echo "ERROR: ${DATA_FILE} missing"; exit 1; }
test -f "/mnt/data/tombench_eval_subset500.jsonl" || { echo "ERROR: subset500 missing"; exit 1; }

# Install ROLL in editable mode (idempotent)
pip install -e /workspace/framework/ROLL >/dev/null

# Run training
cd /workspace/framework/ROLL
exec python examples/start_rlvr_pipeline.py \
  --config_path "${CONFIG_DIR}" \
  --config_name "${CONFIG_NAME}"
