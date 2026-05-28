#!/usr/bin/env bash
# Poll all 8 vLLM endpoints until /v1/models returns 200.
# Run ON the server (after 01_serve_up.sh) — vLLM cold-start takes ~60-90s for 8B.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.191"

# shellcheck disable=SC1090
source "$ENV_FILE"

PORTS="$BASE_PORTS $V10_PORTS"
MAX_WAIT=600   # 10 min
INTERVAL=10

start=$(date +%s)
for port in $PORTS; do
  echo -n "Waiting for port $port ... "
  while true; do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "READY"
      break
    fi
    now=$(date +%s)
    if [ $((now - start)) -gt $MAX_WAIT ]; then
      echo "TIMEOUT after ${MAX_WAIT}s"
      docker logs --tail 40 "qwen3-eval-base-$((port-8001))" 2>/dev/null || true
      docker logs --tail 40 "qwen3-eval-v10-$((port-8005))"  2>/dev/null || true
      exit 1
    fi
    sleep $INTERVAL
  done
done

echo ""
echo "=== All 8 endpoints ready ==="
for port in $PORTS; do
  m=$(curl -s "http://127.0.0.1:${port}/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
  echo "  :$port  $m"
done
