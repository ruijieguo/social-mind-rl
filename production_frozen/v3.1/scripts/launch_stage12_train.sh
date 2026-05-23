#!/usr/bin/env bash
# Track E: Stage 12 training
# Merges stage 8 base + Track B (ExploreToM v2) + Track C (HOT GPT-5.5 synth)
# into tom_train_stage12.jsonl, syncs to TRAIN, then continues training from stage 8.
#
# Prerequisites:
#   - Track D training has completed (releases 8 GPUs)
#   - Track C synthesis has completed (data/tom/raw/synth_gpt55_phase_d_hot.jsonl ≥ 1000 records)
#   - Track B data exists (data/tom/raw/exploretom_v2.jsonl, 2000 records)
set -e
source configs/deploy.env

# Step 1: merge data on DEV
echo "=== Merging stage 12 training data ==="
python3 scripts/data/merge_stage11_train.py \
  --base data/tom/tom_train.jsonl \
  --add data/tom/raw/exploretom_v2.jsonl data/tom/raw/synth_gpt55_phase_d_hot.jsonl \
  --output data/tom/tom_train_stage12.jsonl \
  --eval-data data/tom/tombench_eval.jsonl

echo
echo "=== Syncing data to TRAIN ==="
rsync -av -e "ssh -i $TRAIN_SSH_KEY" \
  data/tom/tom_train_stage12.jsonl \
  $TRAIN_HOST:$TRAIN_DATA_DIR/

echo
echo "=== Launching stage 12 training ==="
LOG="logs/train_stage12_1x8_14b_$(date +%Y%m%d_%H%M%S).log"
nohup ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd $TRAIN_PATH && \
  docker compose -f docker/train/docker-compose.yml \
    --env-file configs/deploy.env \
    run --rm --build \
    -e STAGE=stage12_1x8_14b \
    train
" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/stage12_pid
echo "$LOG" > /tmp/stage12_log
echo "Stage 12 PID $PID, log $LOG"
echo "Tail: tail -f $LOG"
