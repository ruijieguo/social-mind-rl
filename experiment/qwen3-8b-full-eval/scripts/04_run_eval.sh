#!/usr/bin/env bash
# Run the full 3-model x 2-benchmark x 3-protocol matrix.
#
# Run ON the server (172.16.120.191), AFTER 01_serve_up.sh + 02_wait_ready.sh.
# Local vLLM endpoints are hit via 127.0.0.1; DashScope hits the public API.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.191"

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "WARN: DASHSCOPE_API_KEY not set in this shell. DashScope run will fail." >&2
fi
if [[ -z "${DASHSCOPE_BASE_URL:-}" ]]; then
  export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
fi

OUT_DIR="$EXP_ROOT/output"
LOG_DIR="$EXP_ROOT/logs"
# v2 cache: separate from v1 to avoid collisions with the old (max_tokens=4096,
# ZH prefix bug, no reasoning_content collection) responses. Caller can override
# with CACHE_SUFFIX="" to reuse v1 cache when fixes don't apply.
CACHE_SUFFIX="${CACHE_SUFFIX:-_v2}"
CACHE_DIR="$OUT_DIR/cache${CACHE_SUFFIX}"
mkdir -p "$OUT_DIR/tombench" "$OUT_DIR/hitom" "$LOG_DIR" "$CACHE_DIR"

# Container-side paths (everything inside /work because we mount EXP_ROOT to /work)
OUT_DIR_C=/work/output
CACHE_DIR_C="/work/output/cache${CACHE_SUFFIX}"

PROTOCOLS="${PROTOCOLS:-direct,direct_think,cot}"
LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_FLAG="--limit $LIMIT"
fi

# Data paths inside container (mapped from REMOTE_BASE/data)
EVAL_TOMBENCH_C=/data/tom/tombench_eval.jsonl
EVAL_HITOM_C=/data/eval/hitom_eval.jsonl

BASE_EPS=""; for p in $BASE_PORTS; do BASE_EPS="$BASE_EPS 127.0.0.1:$p"; done
V10_EPS="";  for p in $V10_PORTS;  do V10_EPS="$V10_EPS 127.0.0.1:$p"; done

PY="${PYTHON:-python3}"
RUNNER="$EXP_ROOT/scripts/parallel_eval.py"

# If host python lacks openai (the 172.16.120.191 host has no pip), run the
# evaluation client inside the vllm-openai image (which already ships openai +
# tqdm). Set USE_DOCKER_CLIENT=0 to force the host python instead.
USE_DOCKER_CLIENT="${USE_DOCKER_CLIENT:-1}"
CLIENT_IMAGE="${CLIENT_IMAGE:-vllm/vllm-openai:v0.11.0}"

# Wrapper that runs the runner either in-host or in a one-shot container.
# Containers use --network host so they reach 127.0.0.1:8001..8008 + the public
# DashScope endpoint, and -v mounts the experiment dir at the same absolute path.
docker_py() {
  docker run --rm --network host \
    --user "$(id -u):$(id -g)" \
    -e DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}" \
    -e DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
    -e PYTHONPATH=/work/scripts \
    -e HOME=/tmp \
    -v "$EXP_ROOT":/work \
    -v "$REMOTE_BASE/data":/data:ro \
    -w /work \
    --entrypoint=python3 \
    "$CLIENT_IMAGE" "$@"
}

# Function: run one (model, benchmark) pair through 3 protocols
run_set() {
  local backend="$1" mname="$2" mid="$3" bench="$4" data="$5" out="$6" eps="$7" conc="$8"
  local logf="$LOG_DIR/run_${mid}_${bench}.log"
  echo "=== $mid / $bench / $PROTOCOLS  (backend=$backend, conc=$conc) ==="
  echo "    log: $logf"

  local args=(
    --backend "$backend"
    --model "$mname"
    --model-id "$mid"
    --benchmark "$bench"
    --data "$data"
    --protocols "$PROTOCOLS"
    --output "$out"
    --cache-dir "$CACHE_DIR_C"
    --concurrency "$conc"
  )
  if [[ "$backend" == "local" ]]; then
    # shellcheck disable=SC2206
    args+=(--endpoints $eps)
  fi
  if [[ -n "$LIMIT_FLAG" ]]; then
    # shellcheck disable=SC2206
    args+=($LIMIT_FLAG)
  fi

  if [[ "$USE_DOCKER_CLIENT" == "1" ]]; then
    docker_py /work/scripts/parallel_eval.py "${args[@]}" 2>&1 | tee "$logf"
  else
    "$PY" "$RUNNER" "${args[@]}" 2>&1 | tee "$logf"
  fi
}

# Phase selectors:
#   LOCAL_ONLY=1  → run only local vLLM (base + v1.0)
#   DS_ONLY=1     → run only DashScope API
#   (neither)     → run all
LOCAL_ONLY="${LOCAL_ONLY:-0}"
DS_ONLY="${DS_ONLY:-0}"

# Output suffix: v2 results go to base_v2.json / v10_v2.json / dashscope_v2.json
# so v1 results remain on disk for cross-version comparison.
OUT_SUFFIX="${OUT_SUFFIX:-_v2}"

if [[ "$DS_ONLY" != "1" ]]; then
  # ----- Local vLLM: base + v1.0 in parallel (independent GPUs) -----
  run_set local "$BASE_MODEL_NAME" "qwen3-8b-base" tombench "$EVAL_TOMBENCH_C" "$OUT_DIR_C/tombench/base${OUT_SUFFIX}.json" "$BASE_EPS" 32 &
  PID_BASE_TOM=$!
  run_set local "$V10_MODEL_NAME"  "qwen3-8b-v10"  tombench "$EVAL_TOMBENCH_C" "$OUT_DIR_C/tombench/v10${OUT_SUFFIX}.json"  "$V10_EPS"  32 &
  PID_V10_TOM=$!

  wait $PID_BASE_TOM
  wait $PID_V10_TOM

  run_set local "$BASE_MODEL_NAME" "qwen3-8b-base" hitom "$EVAL_HITOM_C" "$OUT_DIR_C/hitom/base${OUT_SUFFIX}.json" "$BASE_EPS" 32 &
  PID_BASE_HIT=$!
  run_set local "$V10_MODEL_NAME"  "qwen3-8b-v10"  hitom "$EVAL_HITOM_C" "$OUT_DIR_C/hitom/v10${OUT_SUFFIX}.json"  "$V10_EPS"  32 &
  PID_V10_HIT=$!

  wait $PID_BASE_HIT
  wait $PID_V10_HIT
fi

if [[ "$LOCAL_ONLY" != "1" ]]; then
  # ----- DashScope: qwen3-8b API. -----
  DS_CONC="${DASHSCOPE_CONCURRENCY:-8}"
  run_set dashscope "$DASHSCOPE_MODEL_NAME" "$DASHSCOPE_MODEL_ID" tombench "$EVAL_TOMBENCH_C" "$OUT_DIR_C/tombench/dashscope${OUT_SUFFIX}.json" "" "$DS_CONC"
  run_set dashscope "$DASHSCOPE_MODEL_NAME" "$DASHSCOPE_MODEL_ID" hitom    "$EVAL_HITOM_C"    "$OUT_DIR_C/hitom/dashscope${OUT_SUFFIX}.json"    "" "$DS_CONC"
fi

echo ""
echo "=== Eval phase done (LOCAL_ONLY=$LOCAL_ONLY, DS_ONLY=$DS_ONLY) ==="
ls -la "$OUT_DIR/tombench" "$OUT_DIR/hitom"
