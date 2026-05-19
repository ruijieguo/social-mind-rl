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

# Detect SFT vs RLVR pipeline by stage name
# SFT stages: sft_stage9_14b, sft_stage9_8b, etc.
# RLVR stages: stage1, stage9_1x8_14b, etc.
if [[ "${STAGE}" == sft_* ]]; then
  PIPELINE="sft"
  CONFIG_NAME="${STAGE#sft_}"
  CONFIG_NAME="sft_config_${CONFIG_NAME}"
else
  PIPELINE="rlvr"
  CONFIG_NAME="rlvr_config_${STAGE}"
fi

echo "=========================================="
echo "TRAIN entrypoint"
echo "  stage:    ${STAGE}"
echo "  pipeline: ${PIPELINE}"
echo "  config:   ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "  CUDA visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo "  GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================="

# Verify data + models present
case "${STAGE}" in
  stage1|stage1_1x8|stage1_2x8|stage1_1x8_14b)
    DATA_FILE="/mnt/data/tom_train_4k.jsonl"
    ;;
  stage6|stage6_1x8|stage6_1x8_14b|stage7|stage7_1x8|stage7_1x8_14b|stage8|stage8_1x8|stage8_1x8_14b|stage9|stage9_1x8|stage9_1x8_14b|stage10|stage10_1x8|stage10_1x8_14b)
    # Stage 6/7/8/9/10 use the FULL cleaned tom_train.jsonl (post-audit + GPT-5.5 synth)
    DATA_FILE="/mnt/data/tom_train.jsonl"
    ;;
  sft_stage9_14b|sft_stage9_8b)
    # SFT stages use the GPT-5.5 reasoning traces dataset
    DATA_FILE="/mnt/data/tom_train_sft.jsonl"
    ;;
  stage2|stage2_1x8|stage2_2x8|stage3_1x8|stage3_l3)
    DATA_FILE="/mnt/data/tom_train.jsonl"
    ;;
  stage4|stage4_1x8|stage5|stage5_1x8)
    DATA_FILE="/mnt/data/tom_train.jsonl"
    ;;
  *)
    echo "ERROR: unknown STAGE=${STAGE}"
    exit 1
    ;;
esac
test -f "${DATA_FILE}" || { echo "ERROR: ${DATA_FILE} missing"; exit 1; }
if [[ "${PIPELINE}" == "rlvr" ]]; then
  test -f "/mnt/data/tombench_eval_subset500.jsonl" || { echo "ERROR: subset500 missing"; exit 1; }
fi

cd /workspace/framework/ROLL
ln -sfn /workspace/configs/tombench-rlvr examples/tombench_configs

if [[ "${PIPELINE}" == "sft" ]]; then
  exec python examples/start_sft_pipeline.py \
    --config_path "tombench_configs" \
    --config_name "${CONFIG_NAME}"
else
  exec python examples/start_rlvr_pipeline.py \
    --config_path "tombench_configs" \
    --config_name "${CONFIG_NAME}"
fi
