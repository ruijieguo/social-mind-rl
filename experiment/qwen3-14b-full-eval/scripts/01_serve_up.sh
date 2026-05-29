#!/usr/bin/env bash
# Bring up 4 vLLM instances (TP=1, GPUs 4-7) serving ONE model.
# Usage:  bash scripts/01_serve_up.sh <base|v35|v31>
# Run ON the host (172.16.120.181) inside experiment/qwen3-14b-full-eval/.
set -euo pipefail

MODEL_KEY="${1:?usage: 01_serve_up.sh <base|v35|v31>}"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.181"
COMPOSE="$EXP_ROOT/docker/docker-compose.yml"

# shellcheck disable=SC1090
source "$ENV_FILE"

case "$MODEL_KEY" in
  base) MP="$BASE_MODEL_PATH"; SN="$BASE_MODEL_NAME" ;;
  v35)  MP="$V35_MODEL_PATH";  SN="$V35_MODEL_NAME"  ;;
  v31)  MP="$V31_MODEL_PATH";  SN="$V31_MODEL_NAME"  ;;
  *) echo "unknown model key: $MODEL_KEY"; exit 1 ;;
esac

test -d "$MP" || { echo "ERROR: model dir $MP missing on host"; exit 1; }

echo "=== Serving $MODEL_KEY ($SN) from $MP on GPUs 4-7 (ports $PORTS) ==="
export MODEL_PATH="$MP" SERVED_NAME="$SN" MODEL_HOST_DIR MAX_MODEL_LEN GPU_UTIL SERVE_PORT_BASE

cd "$EXP_ROOT"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" ps
echo "Next: scripts/02_wait_ready.sh"
