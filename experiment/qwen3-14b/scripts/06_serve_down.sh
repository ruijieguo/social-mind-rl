#!/usr/bin/env bash
# Stop & remove the 4 vLLM instances (frees GPUs 4-7).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.181"
docker compose --env-file "$ENV_FILE" -f "$EXP_ROOT/docker/docker-compose.yml" down
echo "=== eval containers stopped, GPUs 4-7 released ==="
