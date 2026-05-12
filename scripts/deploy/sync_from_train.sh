#!/usr/bin/env bash
set -euo pipefail

# Sync TRAIN → DEV: best checkpoint + tensorboard logs + eval results
if [ ! -f configs/deploy.env ]; then
  echo "ERROR: configs/deploy.env missing"
  exit 1
fi
source configs/deploy.env

mkdir -p output/checkpoints output/tensorboard output/eval

echo "[sync-down] best checkpoint ..."
rsync -avz --progress \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  "${TRAIN_HOST}:${TRAIN_OUTPUT_DIR}/best_checkpoint/" \
  ./output/checkpoints/best/ || echo "(no best_checkpoint yet)"

echo "[sync-down] tensorboard logs ..."
rsync -avz \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  "${TRAIN_HOST}:${TRAIN_OUTPUT_DIR}/tensorboard/" \
  ./output/tensorboard/ || echo "(no tb logs)"

echo "[sync-down] eval results ..."
rsync -avz \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  "${TRAIN_HOST}:${TRAIN_OUTPUT_DIR}/eval/" \
  ./output/eval/ || echo "(no eval results)"

echo "[sync-down] done"
