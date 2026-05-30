#!/usr/bin/env bash
# Evaluate the base⊕v3.1 soups on 8 GPUs. For each soup tag: serve (8×TP=1) →
# wait → eval 4 benchmarks × 3 protocols → down. Same prompts/params as the main
# run, so soup columns drop straight into the report.
#
#   bash scripts/run_soup_eval.sh                 # tags 25 50 75
#   SOUP_TAGS="50" LIMIT=10 bash scripts/run_soup_eval.sh   # smoke
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.181"
COMPOSE="$EXP_ROOT/docker/docker-compose.yml"
# shellcheck disable=SC1090
source "$ENV_FILE"

SOUP_TAGS="${SOUP_TAGS:-25 50 75}"
BENCHES="${BENCHES:-tombench hitom socialiqa emobench}"
PROTOCOLS="${PROTOCOLS:-direct,direct_think,cot}"
CONC="${CONCURRENCY:-128}"
CLIENT_IMAGE="${CLIENT_IMAGE:-qwen3-tom-serve-eval-dp4:latest}"
SOUP_ROOT="${SOUP_ROOT:-/data_nvme/grj-projects/models}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

OUT_DIR="$EXP_ROOT/output"; LOG_DIR="$EXP_ROOT/logs"; CACHE_DIR="$OUT_DIR/cache"
mkdir -p "$LOG_DIR" "$CACHE_DIR"; for b in $BENCHES; do mkdir -p "$OUT_DIR/$b"; done
ENDPOINTS=""; for p in $PORTS; do ENDPOINTS="$ENDPOINTS 127.0.0.1:$p"; done

bench_data(){ case "$1" in
  tombench) echo "$EVAL_TOMBENCH";; hitom) echo "$EVAL_HITOM";;
  socialiqa) echo "$EVAL_SOCIALIQA";; emobench) echo "$EVAL_EMOBENCH";; esac; }

docker_py(){ docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e PYTHONPATH="$EXP_ROOT/scripts" -e HOME=/tmp \
  -v "$REMOTE_BASE":"$REMOTE_BASE" -w "$EXP_ROOT" \
  --entrypoint python3 "$CLIENT_IMAGE" "$@"; }

for tag in $SOUP_TAGS; do
  mid="soup${tag}"; mname="qwen3-14b-soup${tag}"
  mp="$SOUP_ROOT/Qwen3-14B-soup${tag}"
  if [ ! -d "$mp" ]; then echo "SKIP $mid: $mp missing"; continue; fi
  echo "================= SOUP $tag ($mp) ================="
  export MODEL_PATH="$mp" SERVED_NAME="$mname" MODEL_HOST_DIR MAX_MODEL_LEN GPU_UTIL SERVE_PORT_BASE
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d
  bash "$HERE/02_wait_ready.sh"
  for b in $BENCHES; do
    data="$(bench_data "$b")"; out="$OUT_DIR/$b/${mid}.json"; logf="$LOG_DIR/run_${mid}_${b}.log"
    echo "----- $mid / $b / $PROTOCOLS -----"
    # shellcheck disable=SC2086
    docker_py "$EXP_ROOT/scripts/parallel_eval.py" \
      --model "$mname" --model-id "$mid" --endpoints $ENDPOINTS \
      --benchmark "$b" --data "$data" --protocols "$PROTOCOLS" \
      --output "$out" --cache-dir "$CACHE_DIR" --concurrency "$CONC" $LIMIT_FLAG 2>&1 | tee "$logf"
  done
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE" down
done
echo "=== SOUP_EVAL_DONE $(date -Iseconds) ==="
