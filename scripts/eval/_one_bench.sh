#!/usr/bin/env bash
# Run one benchmark against one vLLM endpoint.
#
# Usage: _one_bench.sh <benchmark> <port>
#   benchmark: tombench | emobench | hitom | socialiqa
#   port:      OpenAI-compatible vLLM port on TRAIN_HOST
#
# Reads:  TRAIN_HOST_HOSTONLY (e.g. 172.16.120.191), MODEL_NAME, OUTPUT_DIR
# Writes: $OUTPUT_DIR/<benchmark>.json  (+ .md for tombench)
#
# Three protocols (direct/cot/del_tom) are run sequentially inside the
# runner — that's intentional, since each benchmark already saturates a
# 2-GPU vLLM instance with concurrency=16.

set -euo pipefail

BENCH="$1"
PORT="$2"

: "${TRAIN_HOST_HOSTONLY:?required (e.g. 172.16.120.191)}"
: "${MODEL_NAME:?required (e.g. qwen3.6-27b-base)}"
: "${OUTPUT_DIR:?required}"

CONCURRENCY="${CONCURRENCY:-16}"
DEL_TOM_N="${DEL_TOM_N:-8}"
PROTOCOLS="${PROTOCOLS:-direct,cot,del_tom}"

BASE_URL="http://${TRAIN_HOST_HOSTONLY}:${PORT}/v1"
mkdir -p "$OUTPUT_DIR"

case "$BENCH" in
  tombench)
    RUNNER="scripts/eval/run_tombench.py"
    DATA="data/tom/tombench_eval.jsonl"
    OUT="$OUTPUT_DIR/tombench.json"
    # Cache namespaced by model+protocol to avoid cross-protocol pollution
    # (e.g. unified_v1 thinking-direct should not be reused by unified_v2
    # no-thinking direct).
    EXTRA="--cache-dir output/eval_cache_${MODEL_NAME}_unified_v2"
    ;;
  emobench)
    RUNNER="scripts/eval/run_generic_mcq.py"
    DATA="data/eval/emobench_eval.jsonl"
    OUT="$OUTPUT_DIR/emobench.json"
    EXTRA=""
    ;;
  hitom)
    RUNNER="scripts/eval/run_generic_mcq.py"
    DATA="data/eval/hitom_eval.jsonl"
    OUT="$OUTPUT_DIR/hitom.json"
    EXTRA=""
    ;;
  socialiqa)
    RUNNER="scripts/eval/run_generic_mcq.py"
    DATA="data/eval/socialiqa_eval.jsonl"
    OUT="$OUTPUT_DIR/socialiqa.json"
    EXTRA=""
    ;;
  *)
    echo "ERROR: unknown benchmark '$BENCH'" >&2
    exit 2
    ;;
esac

echo "=========================================="
echo "$BENCH  ->  $BASE_URL"
echo "  data:       $DATA"
echo "  model:      $MODEL_NAME"
echo "  protocols:  $PROTOCOLS"
echo "  concurrency:$CONCURRENCY"
echo "  output:     $OUT"
echo "=========================================="

# Use the dev docker image so dependencies are stable. The container
# inherits OPENAI_API_KEY (vLLM accepts any token) and DATA mounts via
# the existing docker/dev/docker-compose.yml.
exec docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
  dev python "$RUNNER" \
    --backend openai \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --protocols "$PROTOCOLS" \
    --data "$DATA" \
    --output "$OUT" \
    --concurrency "$CONCURRENCY" \
    --del-tom-n "$DEL_TOM_N" \
    ${LIMIT:+--limit "$LIMIT"} \
    $EXTRA
