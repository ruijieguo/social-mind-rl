#!/usr/bin/env bash
# Stage 13: 续训 Stage 12 (no new data, just more steps from Stage 12 ckpt)
#
# Prerequisites:
#   - Stage 12 HF model at /data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf-stage12/
#   - 8 GPUs free on TRAIN host (eval-serve-stage12 already stopped)
#   - tom_train_stage12.jsonl already on TRAIN host (from Stage 12 launch)
set -e
source configs/deploy.env

echo "=== Verify Stage 12 HF model exists ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "ls -la $TRAIN_OUTPUT_DIR/qwen3-14B-tom-hf-stage12/config.json"

echo
echo "=== Verify training data on TRAIN ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "ls -la $TRAIN_DATA_DIR/tom_train_stage12.jsonl"

echo
echo "=== Verify GPUs free ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'

echo
echo "=== Launching Stage 13 training (continue from Stage 12) ==="
LOG="logs/train_stage13_1x8_14b_$(date +%Y%m%d_%H%M%S).log"
nohup ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd $TRAIN_PATH && \
  docker compose -f docker/train/docker-compose.yml \
    --env-file configs/deploy.env \
    run --rm --build \
    -e STAGE=stage13_1x8_14b \
    train
" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/stage13_pid
echo "$LOG" > /tmp/stage13_log
echo "Stage 13 PID $PID, log $LOG"
echo "Tail: tail -f $LOG"
