#!/usr/bin/env bash
# Autonomous Plan-A (Stage 22) launcher — runs detached ON .191.
# Honors the user's choice: prep is done; this waits for (a) the base download
# to finish and (b) all 8 GPUs to free, then launches training and verifies the
# reward override took effect. Never disturbs the running job — only waits.
set -uo pipefail

REPO=/home/h800/grj-projects/qwen3-tom
MODELS=/home/h800/grj-projects/models
DATA=/home/h800/grj-projects/tom-data
OUT=/home/h800/grj-projects/tom-output
LOG="$REPO/experiment/qwen3-14b/logs/train_stage22.log"
GPU_FREE_MIB=6000
mkdir -p "$REPO/experiment/qwen3-14b/logs"

echo "[autolaunch] $(date -Iseconds) waiting for base Qwen3-14B download (8 shards)..."
for i in $(seq 1 240); do   # up to ~4h
  n=$(ls "$MODELS/Qwen3-14B"/*.safetensors 2>/dev/null | wc -l)
  idx=$([ -f "$MODELS/Qwen3-14B/model.safetensors.index.json" ] && echo 1 || echo 0)
  dl=$(docker ps -a --filter name=dl-14b-191 --format '{{.Status}}' 2>/dev/null)
  [ "$n" -eq 8 ] && [ "$idx" = 1 ] && echo "[autolaunch] base ready ($n shards)" && break
  echo "[autolaunch] $(date +%H:%M:%S) base: $n/8 shards (dl: ${dl:-gone})"
  sleep 60
done

echo "[autolaunch] $(date -Iseconds) waiting for all 8 GPUs to free (<${GPU_FREE_MIB} MiB)..."
for i in $(seq 1 1440); do  # up to ~24h
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v t="$GPU_FREE_MIB" '$1>t{c++} END{print c+0}')
  [ "$busy" -eq 0 ] && echo "[autolaunch] GPUs free at $(date -Iseconds)" && break
  echo "[autolaunch] $(date +%H:%M:%S) $busy/8 GPUs still busy"
  sleep 120
done

echo "[autolaunch] launching Stage 22 $(date -Iseconds)"
cd "$REPO"
STAGE=stage22_qwen3_14b \
TRAIN_DATA_DIR="$DATA" \
TRAIN_MODELS_DIR="$MODELS" \
TRAIN_OUTPUT_DIR="$OUT" \
  docker compose -f docker/train/docker-compose.yml up -d

echo "[autolaunch] launched; waiting for reward-override + first step (up to ~8 min)..."
ok=0
for i in $(seq 1 48); do
  sleep 10
  docker compose -f docker/train/docker-compose.yml logs --no-color 2>/dev/null | tail -400 > "$LOG.tmp" || true
  if grep -q '\[tom_mcq_reward\] resolved' "$LOG.tmp"; then ok=1; break; fi
  # surface fatal errors early
  if grep -qiE "Traceback|CUDA out of memory|ERROR: " "$LOG.tmp"; then echo "[autolaunch] EARLY ERROR detected"; break; fi
done
echo "=== reward override line ==="; grep '\[tom_mcq_reward\]' "$LOG.tmp" | tail -3
echo "=== container status ==="; docker compose -f docker/train/docker-compose.yml ps
echo "=== GPU ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
if [ "$ok" = 1 ]; then echo "[autolaunch] STAGE22_LAUNCHED_OK $(date -Iseconds)"; else echo "[autolaunch] STAGE22_LAUNCH_UNVERIFIED — inspect $LOG"; fi
docker compose -f docker/train/docker-compose.yml logs --no-color 2>/dev/null > "$LOG" || true
