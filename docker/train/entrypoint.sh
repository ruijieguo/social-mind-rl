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

# Route 27B stages to the dedicated experiment/qwen3.6-27b/configs/ tree;
# everything else uses the project-wide configs/tombench-rlvr/. This keeps
# the 27B work isolated from the 14B/8B production configs.
if [[ "${STAGE}" == *_27b* ]]; then
  CONFIG_DIR="/workspace/experiment/qwen3.6-27b/configs"
else
  CONFIG_DIR="/workspace/configs/tombench-rlvr"
fi

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
  stage6|stage6_1x8|stage6_1x8_14b|stage7|stage7_1x8|stage7_1x8_14b|stage8|stage8_1x8|stage8_1x8_14b|stage9|stage9_1x8|stage9_1x8_14b|stage10|stage10_1x8|stage10_1x8_14b|stage11d_continue_1x8_14b)
    # Stage 6/7/8/9/10/11d use the FULL cleaned tom_train.jsonl (post-audit + GPT-5.5 synth)
    DATA_FILE="/mnt/data/tom_train.jsonl"
    ;;
  stage12_1x8_14b)
    # Stage 12: stage 8 base + Track B (ExploreToM v2) + Track C (HOT synth)
    DATA_FILE="/mnt/data/tom_train_stage12.jsonl"
    ;;
  stage13_1x8_14b)
    # Stage 13: continue from Stage 12 ckpt, same data (testing if Stage 12 plateaued)
    DATA_FILE="/mnt/data/tom_train_stage12.jsonl"
    ;;
  stage14_1x8_14b)
    # Stage 14: task-weighted resample of Stage 12 data (Knowledge ×1.92, FB ×0.72)
    DATA_FILE="/mnt/data/tom_train_stage14_weighted.jsonl"
    ;;
  stage14b_1x8_8b)
    # Stage 14b 8B: same task-weighted data on Qwen3-8B from Stage 7 ckpt
    DATA_FILE="/mnt/data/tom_train_stage14b_weighted.jsonl"
    ;;
  stage15_1x8_8b)
    # Stage 15 8B: filter out reward>=0.95 records, re-weight by 8B Stage 7's
    # own per-task del_tom acc (Knowledge×2.0, FB×0.77). 7482 records.
    DATA_FILE="/mnt/data/tom_train_stage15_8b_filtered_weighted.jsonl"
    ;;
  stage16_1x8_14b)
    # Stage 16 14B: continue from v3.1 (Stage 14b ckpt-199) with Hi-ToM (1200) +
    # EmoBench EU_emotion/EU_cause/EA + SocialIQA synth (1500) added on top of
    # the Stage 14 weighted backbone. Targets the structural data gaps that
    # caused +20-42pp v3.1-vs-DeepSeek gaps on Hi-ToM and EU_emotion.
    DATA_FILE="/mnt/data/tom_train_stage16.jsonl"
    test -f /mnt/data/hitom_eval_val200.jsonl \
      || { echo "ERROR: /mnt/data/hitom_eval_val200.jsonl missing"; exit 1; }
    test -f /mnt/data/emobench_eu_emotion_val100.jsonl \
      || { echo "ERROR: /mnt/data/emobench_eu_emotion_val100.jsonl missing"; exit 1; }
    ;;
  stage17_1x8_14b)
    # Stage 17 14B: v3.3 = continue from v3.2 (Stage 16 ckpt-270) with targeted
    # gap-closing data: Hi-ToM direct-style (960), EU_emotion expansion (1500),
    # EU_cause/EA/SocialIQA increments (1400), Belief distillation (~600).
    DATA_FILE="/mnt/data/tom_train_stage17.jsonl"
    test -f /mnt/data/hitom_eval_val200.jsonl \
      || { echo "ERROR: /mnt/data/hitom_eval_val200.jsonl missing"; exit 1; }
    test -f /mnt/data/emobench_eu_emotion_val100.jsonl \
      || { echo "ERROR: /mnt/data/emobench_eu_emotion_val100.jsonl missing"; exit 1; }
    ;;
  stage18_1x8_14b)
    # Stage 18 14B: v3.4 = continue from v3.3 (Stage 17 ckpt-120) with GPT-5.5
    # distillation of v3.3 actual eval errors. ~734 distilled records added on
    # top of stage17 backbone (~19033 total).
    DATA_FILE="/mnt/data/tom_train_stage18.jsonl"
    test -f /mnt/data/hitom_eval_val200.jsonl \
      || { echo "ERROR: /mnt/data/hitom_eval_val200.jsonl missing"; exit 1; }
    test -f /mnt/data/emobench_eu_emotion_val100.jsonl \
      || { echo "ERROR: /mnt/data/emobench_eu_emotion_val100.jsonl missing"; exit 1; }
    ;;
  stage19_1x8_14b)
    # Stage 19 14B: v3.5 = improved GPT-5.5 distillation (3-sample voting +
    # emotion ontology injection). Init from v3.3 ckpt-120 (NOT v3.4) — fresh
    # test of pure improved-distill effect. ~450 high-quality distill records.
    DATA_FILE="/mnt/data/tom_train_stage19.jsonl"
    test -f /mnt/data/hitom_eval_val200.jsonl \
      || { echo "ERROR: /mnt/data/hitom_eval_val200.jsonl missing"; exit 1; }
    test -f /mnt/data/emobench_eu_emotion_val100.jsonl \
      || { echo "ERROR: /mnt/data/emobench_eu_emotion_val100.jsonl missing"; exit 1; }
    ;;
  stage1_27b_1x8|stage1_27b_1x8_smoke)
    # Stage 1 27B (v4.0): mirror of 14B Stage 16 (v3.2) recipe on Qwen3.6-27B
    # base. TP=4 (27B in TP=2 OOMs on 80GB H800). Same data + validation
    # triplet as 14B Stage 16. _smoke variant runs 3 steps for pre-flight.
    DATA_FILE="/mnt/data/tom_train_stage16.jsonl"
    test -f /mnt/data/hitom_eval_val200.jsonl \
      || { echo "ERROR: /mnt/data/hitom_eval_val200.jsonl missing"; exit 1; }
    test -f /mnt/data/emobench_eu_emotion_val100.jsonl \
      || { echo "ERROR: /mnt/data/emobench_eu_emotion_val100.jsonl missing"; exit 1; }
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
ln -sfn "${CONFIG_DIR}" examples/tombench_configs

if [[ "${PIPELINE}" == "sft" ]]; then
  exec python examples/start_sft_pipeline.py \
    --config_path "tombench_configs" \
    --config_name "${CONFIG_NAME}"
else
  exec python examples/start_rlvr_pipeline.py \
    --config_path "tombench_configs" \
    --config_name "${CONFIG_NAME}"
fi
