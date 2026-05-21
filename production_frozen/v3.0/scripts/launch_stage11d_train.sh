#!/usr/bin/env bash
# Launch stage 11D training (continue from stage 8) on TRAIN host.
# Prerequisite: vLLM serve on port 8000 must be stopped (frees GPU 0).
set -e
source configs/deploy.env

ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" '
  docker stop eval-serve-stage8 2>/dev/null || true
  docker rm eval-serve-stage8 2>/dev/null || true
  sleep 5
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
'

LOG="logs/train_stage11d_1x8_14b_$(date +%Y%m%d_%H%M%S).log"
echo "Launching stage 11D, log: $LOG"

ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd $TRAIN_PATH && \
  docker compose -f docker/train/docker-compose.yml \
    --env-file configs/deploy.env \
    run --rm --build \
    -e STAGE=stage11d_continue_1x8_14b \
    train
" > "$LOG" 2>&1 &
echo "$!" > /tmp/stage11d_pid
echo "PID $(cat /tmp/stage11d_pid)"
echo "Tail: tail -f $LOG"
echo "$LOG" > /tmp/stage11d_log
