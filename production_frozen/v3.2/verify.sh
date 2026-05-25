#!/usr/bin/env bash
# verify.sh — verify production_frozen/v3.2 artifacts.
#
# Local checks:
#   1. All files in v3.2/ tree match SHA256SUMS.txt
#
# Remote checks (require ssh access to TRAIN host h800@172.16.120.181):
#   2. The HF model at /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270
#      matches the SHA256 of each safetensor + tokenizer file.
#
# Usage:
#   cd production_frozen/v3.2
#   ./verify.sh                # local files only
#   ./verify.sh --remote       # also verify remote HF model
#
set -euo pipefail

cd "$(dirname "$0")"
SUMS="SHA256SUMS.txt"

echo "[1/2] Verifying local files in $(pwd) ..."
LOCAL_BLOCK=$(awk '/^# Remote/{exit} {print}' "$SUMS")
echo "$LOCAL_BLOCK" | shasum -a 256 -c -
echo "OK: local files match"

if [ "${1:-}" = "--remote" ]; then
  TRAIN_HOST="${TRAIN_HOST:-h800@172.16.120.181}"
  TRAIN_KEY="${TRAIN_SSH_KEY:-$HOME/.ssh/id_ed25519}"
  REMOTE_DIR="/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270"
  echo
  echo "[2/2] Verifying remote HF model at ${TRAIN_HOST}:${REMOTE_DIR}"
  REMOTE_BLOCK=$(awk '/^# Remote/{flag=1; next} flag {print}' "$SUMS" | sed 's|qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270/||')
  diff <(echo "$REMOTE_BLOCK" | sed -E 's/^[[:space:]]+//' | sort) \
       <(ssh -i "$TRAIN_KEY" "$TRAIN_HOST" "cd $REMOTE_DIR && sha256sum *.safetensors *.json *.txt *.jinja" 2>/dev/null | sort) \
    && echo "OK: remote HF model matches" \
    || { echo "FAIL: remote HF model SHA256 mismatch"; exit 1; }
fi

echo
echo "All checks passed."
