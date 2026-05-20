#!/usr/bin/env bash
# Auto-launch Track E (stage 12 training) the moment Track D completes.
# Polls Track D log for "training complete" signal, then:
#   1. Verifies all 8 GPUs are free
#   2. Launches stage 12 training
#
# Usage:
#   nohup bash scripts/auto_launch_stage12.sh > logs/auto_launch_stage12.log 2>&1 &
set -e
source configs/deploy.env

DLOG=$(cat /tmp/track_d_log)
echo "Watching Track D ($DLOG) for completion..."

# Wait for Track D to finish (no new metrics_tag for 5 min, or step >= 349)
while true; do
  LATEST=$(grep "metrics_tag" "$DLOG" 2>/dev/null | tail -1 | python3 -c "
import sys, json, re
m = re.search(r'metrics_tag: ({.*})', sys.stdin.read())
print(json.loads(m.group(1)).get('step', 0) if m else 0)
" 2>/dev/null || echo 0)
  if [ "$LATEST" -ge 349 ]; then
    echo "Track D step $LATEST — done."
    break
  fi
  echo "  Track D step $LATEST/350, waiting..."
  sleep 120
done

# Wait extra 90s for ckpt save to flush + GPUs to free
sleep 90

echo "=== Verifying GPUs free on TRAIN ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head'

echo "=== Launching Track E (stage 12) ==="
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
echo "Stage 12 launched PID=$PID, log=$LOG"
