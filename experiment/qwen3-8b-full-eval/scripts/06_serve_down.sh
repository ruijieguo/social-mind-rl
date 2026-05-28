#!/usr/bin/env bash
# Stop & remove all 8 vLLM instances.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.191"
COMPOSE="$EXP_ROOT/docker/docker-compose.yml"

cd "$EXP_ROOT"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" down
echo "=== All eval containers stopped ==="
