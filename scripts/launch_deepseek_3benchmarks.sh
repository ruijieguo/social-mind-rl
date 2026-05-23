#!/usr/bin/env bash
# Launch deepseek-v4-pro evals across 3 benchmarks × 3 protocols (serial).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  # Try to read from .env
  if [ -f .env ]; then
    DEEPSEEK_API_KEY=$(grep '^DEEPSEEK_API_KEY=' .env | cut -d= -f2-)
  fi
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY not set"
  exit 1
fi
echo "DEEPSEEK_API_KEY length: ${#DEEPSEEK_API_KEY}"

mkdir -p output/eval

for spec in "socialiqa:data/eval/socialiqa_eval.jsonl" \
            "emobench:data/eval/emobench_eval.jsonl" \
            "hitom:data/eval/hitom_eval.jsonl"; do
  BENCH=${spec%:*}
  DATA=${spec##*:}
  echo
  echo "================================================================"
  echo "=== deepseek-v4-pro: $BENCH (direct + cot + del_tom) ==="
  echo "================================================================"
  docker compose -f docker/dev/docker-compose.yml run --rm \
    -e DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
    -v /Users/jaredguo-mini/develop/training:/workspace \
    -w /workspace dev \
    python scripts/eval/run_generic_mcq.py \
      --backend deepseek --model deepseek-v4-pro \
      --data "/workspace/$DATA" \
      --protocols direct,cot,del_tom \
      --concurrency 32 \
      --output "/workspace/output/eval/deepseek_${BENCH}_full.json"
done

echo
echo "All 3 deepseek benchmark evals done."
