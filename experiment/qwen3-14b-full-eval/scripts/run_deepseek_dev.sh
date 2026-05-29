#!/usr/bin/env bash
# Run deepseek-v4-pro (DeepSeek official API) over all 4 benchmarks × 3 protocols
# ON THE DEV MACHINE (mac), inside the qwen3-tom-dev image (ships openai+tqdm).
# Same prompts/extractors/sampling as the local Qwen3-14B run → fair 4th column.
#
#   DEEPSEEK_API_KEY=sk-... bash scripts/run_deepseek_dev.sh
#   LIMIT=10 ... bash scripts/run_deepseek_dev.sh        # smoke test
#
# Reads DEEPSEEK_API_KEY from the environment (export it or source ~/.zshrc first).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"

: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY (it is in ~/.zshrc)}"
IMAGE="${IMAGE:-qwen3-tom-dev:latest}"
PROTOCOLS="${PROTOCOLS:-direct,direct_think,cot}"
CONC="${CONCURRENCY:-12}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"
BENCHES="${BENCHES:-tombench hitom socialiqa emobench}"

# (macOS /bin/bash is 3.2 → no associative arrays; use a case function)
bench_data() {
  case "$1" in
    tombench)  echo data/tom/tombench_eval.jsonl ;;
    hitom)     echo data/eval/hitom_eval.jsonl ;;
    socialiqa) echo data/eval/socialiqa_eval.jsonl ;;
    emobench)  echo data/eval/emobench_eval.jsonl ;;
  esac
}

OUT_REL="experiment/qwen3-14b-full-eval/output"
CACHE_REL="$OUT_REL/cache"
mkdir -p "$REPO_ROOT/$OUT_REL"/{tombench,hitom,socialiqa,emobench} "$REPO_ROOT/$CACHE_REL"

for b in $BENCHES; do
  echo "=== deepseek / $b / $PROTOCOLS ==="
  data_rel="$(bench_data "$b")"
  logf="$EXP_ROOT/logs/run_deepseek_${b}.log"
  docker run --rm -i --network host \
    -e DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
    -e DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}" \
    -e PYTHONPATH=/work/experiment/qwen3-14b-full-eval/scripts \
    -e HOME=/tmp \
    -v "$REPO_ROOT":/work -w /work \
    --entrypoint python3 "$IMAGE" \
    experiment/qwen3-14b-full-eval/scripts/deepseek_eval.py \
      --benchmark "$b" --data "/work/${data_rel}" \
      --protocols "$PROTOCOLS" \
      --output "/work/$OUT_REL/$b/deepseek.json" \
      --cache-dir "/work/$CACHE_REL" \
      --concurrency "$CONC" \
      $LIMIT_FLAG 2>&1 | tee "$logf"
done

echo "=== deepseek DONE $(date -Iseconds) ==="
ls -la "$REPO_ROOT/$OUT_REL"/*/deepseek.json 2>/dev/null
