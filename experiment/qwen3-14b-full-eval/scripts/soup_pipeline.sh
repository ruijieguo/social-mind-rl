#!/usr/bin/env bash
# Autonomous Plan-D pipeline (run detached on the host):
#   1. wait for soup creation (mk-soup container → ALL_SOUPS_DONE)
#   2. wait for GPUs to free (another job holds them; do NOT disturb it)
#   3. eval all soups on 8 GPUs
# Honors the user's choice: WAIT for the foreign GPU job, never co-locate.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_FREE_MIB="${GPU_FREE_MIB:-6000}"   # consider a GPU free if used < this
POLL="${POLL:-300}"

echo "[pipeline] $(date -Iseconds) waiting for soups..."
for i in $(seq 1 240); do
  if docker logs mk-soup 2>&1 | grep -q ALL_SOUPS_DONE; then echo "[pipeline] soups ready"; break; fi
  sleep 60
done

echo "[pipeline] $(date -Iseconds) waiting for all 8 GPUs to free (<${GPU_FREE_MIB} MiB each)..."
for i in $(seq 1 288); do   # up to 24h
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v t="$GPU_FREE_MIB" '$1>t{c++} END{print c+0}')
  if [ "$busy" -eq 0 ]; then echo "[pipeline] GPUs free at $(date -Iseconds)"; break; fi
  echo "[pipeline] $(date +%H:%M:%S) $busy/8 GPUs still busy"
  sleep "$POLL"
done

echo "[pipeline] launching soup eval $(date -Iseconds)"
bash "$HERE/run_soup_eval.sh"
echo "[pipeline] PIPELINE_DONE $(date -Iseconds)"
