#!/usr/bin/env bash
# Run all 4 ToM-family benchmarks (ToMBench / EmoBench / SocialIQA /
# Hi-ToM) against the 4 vLLM instances brought up by
# docker/serve/eval_dp4_compose.yml on the .191 training server.
#
# Each benchmark goes to a dedicated port (saturating one 2-GPU TP=2
# instance), and all 4 run in parallel.
#
# Usage:
#   bash scripts/eval/run_qwen36_27b_baseline.sh           # full run
#   LIMIT=20 bash scripts/eval/run_qwen36_27b_baseline.sh  # smoke test
#
# Output: output/eval/qwen36_27b_baseline/{tombench,emobench,hitom,socialiqa}.json
#         logs/qwen36_27b_<bench>_<timestamp>.log

set -euo pipefail

# Source TRAIN_HOST so we can derive the host-only IP.
if [[ -f configs/deploy.env ]]; then
  set -a; . configs/deploy.env; set +a
fi
TRAIN_HOST_HOSTONLY="${TRAIN_HOST_HOSTONLY:-$(echo "${TRAIN_HOST}" | sed 's/.*@//')}"

export TRAIN_HOST_HOSTONLY
export MODEL_NAME="${MODEL_NAME:-qwen3.6-27b-base}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/eval/qwen36_27b_baseline}"
export PROTOCOLS="${PROTOCOLS:-direct,cot,del_tom}"
export CONCURRENCY="${CONCURRENCY:-16}"
export DEL_TOM_N="${DEL_TOM_N:-8}"

mkdir -p "$OUTPUT_DIR" logs

# Pre-flight: probe each port. Refuse to start unless all 4 are alive.
PORTS=(8001 8002 8003 8004)
echo "Pre-flight: checking 4 vLLM endpoints on $TRAIN_HOST_HOSTONLY ..."
for p in "${PORTS[@]}"; do
  if ! curl -fsS --max-time 5 "http://${TRAIN_HOST_HOSTONLY}:${p}/v1/models" >/dev/null; then
    echo "ERROR: vLLM endpoint :${p} not reachable. Bring up serves first:" >&2
    echo "  ssh ${TRAIN_HOST} 'cd ${TRAIN_PATH} && docker compose -f docker/serve/eval_dp4_compose.yml --env-file configs/deploy.env up -d --build'" >&2
    exit 1
  fi
  echo "  OK :$p"
done

BENCHES=(tombench emobench hitom socialiqa)
TS=$(date +%Y%m%d_%H%M%S)
declare -a PIDS

for i in 0 1 2 3; do
  bench=${BENCHES[$i]}
  port=${PORTS[$i]}
  log="logs/qwen36_27b_${bench}_${TS}.log"
  echo "Launching $bench -> port $port  (log: $log)"
  bash scripts/eval/_one_bench.sh "$bench" "$port" >"$log" 2>&1 &
  PIDS+=($!)
done

echo
echo "All 4 benchmarks launched.  PIDs: ${PIDS[*]}"
echo "Waiting for completion ..."

FAIL=0
for i in 0 1 2 3; do
  bench=${BENCHES[$i]}
  pid=${PIDS[$i]}
  if wait "$pid"; then
    echo "  [OK]   $bench"
  else
    echo "  [FAIL] $bench (pid $pid) — see logs/qwen36_27b_${bench}_${TS}.log"
    FAIL=$((FAIL+1))
  fi
done

if (( FAIL > 0 )); then
  echo "$FAIL benchmark(s) failed."
  exit 1
fi

echo
echo "=== Summary tails ==="
for bench in "${BENCHES[@]}"; do
  log="logs/qwen36_27b_${bench}_${TS}.log"
  echo "--- $bench ---"
  tail -25 "$log" | grep -E "Summary|direct|cot|del_tom|wrote|n=" | head -20 || true
done

echo
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/"
