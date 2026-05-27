#!/usr/bin/env bash
# Bootstrap the .191 training server (h800@172.16.120.191) by pulling
# repo + tom-data from .181 (h800@172.16.120.181) over the internal
# network. .191 has the same NVIDIA driver / docker / 8×H800 stack as
# .181 but a different storage layout:
#
#   .181: /data_nvme/grj-projects/{qwen3-tom,tom-data,models,tom-output}
#   .191: /home/h800/grj-projects/{qwen3-tom,tom-data,models,tom-output}
#         (no /data_nvme mount; sudo requires password; /home is on a
#          private 3.5T NVMe with ~2.6T free)
#
# .191 also has /model/Qwen3-14B and /model/Qwen3-8B already cached
# from another team's work — we DO NOT copy those, since the 27B
# baseline experiment will download Qwen3.6-27B fresh in its own slot
# under /home/h800/grj-projects/models/.
#
# Usage (from DEV macOS, repo root):
#   ssh -i ~/.ssh/id_ed25519 h800@172.16.120.191 'bash -s' \
#     < scripts/deploy/bootstrap_191_from_181.sh
#
# Idempotent: rsync skips up-to-date files. Safe to re-run.

set -euo pipefail

REMOTE_181="h800@172.16.120.181"
ROOT_181="/data_nvme/grj-projects"
ROOT_191="/home/h800/grj-projects"

echo "=========================================="
echo "Bootstrap .191 from $REMOTE_181"
echo "  target root: $ROOT_191"
echo "=========================================="

mkdir -p \
  "$ROOT_191/qwen3-tom" \
  "$ROOT_191/tom-data" \
  "$ROOT_191/tom-output" \
  "$ROOT_191/models"

echo
echo "[1/3] Sync repo (qwen3-tom) from .181 ..."
rsync -aHP \
  --exclude '.git' \
  --exclude 'output/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "$REMOTE_181:$ROOT_181/qwen3-tom/" \
  "$ROOT_191/qwen3-tom/"

echo
echo "[2/3] Sync tom-data from .181 ..."
rsync -aHP \
  "$REMOTE_181:$ROOT_181/tom-data/" \
  "$ROOT_191/tom-data/"

echo
echo "[3/3] Verify ..."
echo "--- disk ---"
df -h "$ROOT_191" || true
echo "--- repo size ---"
du -sh "$ROOT_191/qwen3-tom" || true
echo "--- data files ---"
ls -la "$ROOT_191/tom-data/" | head -30
echo "--- key eval files present? ---"
for f in tombench_eval.jsonl hitom_eval.jsonl emobench_eval.jsonl socialiqa_eval.jsonl; do
  if [[ -f "$ROOT_191/tom-data/$f" ]]; then
    n=$(wc -l < "$ROOT_191/tom-data/$f")
    echo "  OK $f  ($n lines)"
  else
    # tombench / hitom / emobench / socialiqa eval jsonls live in different
    # places on .181: tombench in tom-data, others in repo's data/eval/.
    # Try the repo path as a fallback.
    alt="$ROOT_191/qwen3-tom/data/eval/$f"
    [[ -f "$alt" ]] || alt="$ROOT_191/qwen3-tom/data/tom/$f"
    if [[ -f "$alt" ]]; then
      n=$(wc -l < "$alt")
      echo "  OK $f -> $alt  ($n lines)"
    else
      echo "  MISSING $f  (looked in tom-data and repo/data/{eval,tom})"
    fi
  fi
done
echo "--- gpus ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "--- docker ---"
docker --version
echo
echo "Bootstrap complete."
