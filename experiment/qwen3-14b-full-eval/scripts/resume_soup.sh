#!/usr/bin/env bash
# Robust resume of the soup eval after the driver died at the soup25→soup50
# transition. soup50 servers are already up; eval them in place, then do soup75
# fresh. Deliberately NOT using `set -e` so a cosmetic non-zero can't abort the run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.181"
COMPOSE="$EXP_ROOT/docker/docker-compose.yml"
# shellcheck disable=SC1090
source "$ENV_FILE"

BENCHES="tombench hitom socialiqa emobench"
PROTOCOLS="${PROTOCOLS:-direct,direct_think,cot}"
CONC="${CONCURRENCY:-128}"
CLIENT_IMAGE="qwen3-tom-serve-eval-dp4:latest"
SOUP_ROOT="/data_nvme/grj-projects/models"
OUT_DIR="$EXP_ROOT/output"; LOG_DIR="$EXP_ROOT/logs"; CACHE_DIR="$OUT_DIR/cache"
ENDPOINTS=""; for p in $PORTS; do ENDPOINTS="$ENDPOINTS 127.0.0.1:$p"; done

bench_data(){ case "$1" in tombench) echo "$EVAL_TOMBENCH";; hitom) echo "$EVAL_HITOM";;
  socialiqa) echo "$EVAL_SOCIALIQA";; emobench) echo "$EVAL_EMOBENCH";; esac; }
docker_py(){ docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e PYTHONPATH="$EXP_ROOT/scripts" -e HOME=/tmp -v "$REMOTE_BASE":"$REMOTE_BASE" \
  -w "$EXP_ROOT" --entrypoint python3 "$CLIENT_IMAGE" "$@"; }

eval_one_soup(){
  local tag="$1" mid="soup$1" mname="qwen3-14b-soup$1"
  for b in $BENCHES; do
    echo "----- $mid / $b -----"
    docker_py "$EXP_ROOT/scripts/parallel_eval.py" \
      --model "$mname" --model-id "$mid" --endpoints $ENDPOINTS \
      --benchmark "$b" --data "$(bench_data "$b")" --protocols "$PROTOCOLS" \
      --output "$OUT_DIR/$b/${mid}.json" --cache-dir "$CACHE_DIR" --concurrency "$CONC" \
      2>&1 | tee "$LOG_DIR/run_${mid}_${b}.log"
  done
}

serve_up(){ local tag="$1"
  export MODEL_PATH="$SOUP_ROOT/Qwen3-14B-soup${tag}" SERVED_NAME="qwen3-14b-soup${tag}" \
    MODEL_HOST_DIR MAX_MODEL_LEN GPU_UTIL SERVE_PORT_BASE
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d
  bash "$HERE/02_wait_ready.sh"
}
serve_down(){ docker compose --env-file "$ENV_FILE" -f "$COMPOSE" down; }

# --- soup50: servers already up (serving qwen3-14b-soup50). Confirm, else (re)serve. ---
served="$(curl -s http://127.0.0.1:8001/v1/models 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
echo "=== :8001 currently serving: ${served:-NONE} ==="
if [ "$served" != "qwen3-14b-soup50" ]; then
  echo "=== soup50 not live; bringing it up ==="; serve_down; serve_up 50
fi
echo "================= RESUME SOUP 50 ================="
eval_one_soup 50
serve_down

# --- soup75 fresh ---
echo "================= SOUP 75 ================="
serve_up 75
eval_one_soup 75
serve_down

echo "=== SOUP_EVAL_DONE $(date -Iseconds) ==="
