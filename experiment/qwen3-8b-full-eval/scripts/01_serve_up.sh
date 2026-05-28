#!/usr/bin/env bash
# Bring up 8 vLLM instances on 172.16.120.191 (run THIS script ON the server,
# inside experiment/qwen3-8b-full-eval/).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.191"
COMPOSE="$EXP_ROOT/docker/docker-compose.yml"

echo "=== Bringing up 8 vLLM instances ==="
echo "  env:     $ENV_FILE"
echo "  compose: $COMPOSE"

cd "$EXP_ROOT"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d

echo ""
echo "=== Containers ==="
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" ps

echo ""
echo "Next: run scripts/02_wait_ready.sh to wait until /v1/models responds on all 8 ports."
