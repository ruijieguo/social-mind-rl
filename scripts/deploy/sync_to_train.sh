#!/usr/bin/env bash
set -euo pipefail

# Sync DEV → TRAIN: code + data
# Requires: configs/deploy.env with TRAIN_HOST, TRAIN_PATH, TRAIN_SSH_KEY, TRAIN_DATA_DIR

if [ ! -f configs/deploy.env ]; then
  echo "ERROR: configs/deploy.env missing. Copy from configs/deploy.env.example and fill in."
  exit 1
fi
source configs/deploy.env

# 1. Code + configs (excludes output/data/.git/cache + pip build artefacts).
# pip install -e inside the train container writes *.egg-info/ as root, so
# these can't be deleted/overwritten by rsync running as a non-root user on
# the host. We exclude them entirely — they get regenerated at container
# startup anyway.
echo "[sync-up] syncing code → ${TRAIN_HOST}:${TRAIN_PATH}/"
rsync -avz --delete \
  --exclude=output \
  --exclude=data \
  --exclude=.git \
  --exclude='.cache' \
  --exclude='**/.cache' \
  --exclude='**/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='**/build/' \
  --exclude='**/.DS_Store' \
  --exclude='**/node_modules' \
  --exclude='logs' \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  ./ "${TRAIN_HOST}:${TRAIN_PATH}/"

# 2. Training data (no --delete; manual cleanup if needed)
echo "[sync-up] syncing data → ${TRAIN_HOST}:${TRAIN_DATA_DIR}/"
ssh -i "${TRAIN_SSH_KEY}" "${TRAIN_HOST}" "mkdir -p ${TRAIN_DATA_DIR}"
rsync -avz --progress \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  ./data/tom/ "${TRAIN_HOST}:${TRAIN_DATA_DIR}/"

echo "[sync-up] done"
