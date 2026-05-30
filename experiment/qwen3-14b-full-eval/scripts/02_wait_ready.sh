#!/usr/bin/env bash
# Poll the 4 vLLM endpoints until /v1/models returns 200 (14B cold-start ~90-150s).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.181"
# shellcheck disable=SC1090
source "$ENV_FILE"

MAX_WAIT="${MAX_WAIT:-900}"
INTERVAL=10
start=$(date +%s)

for port in $PORTS; do
  echo -n "Waiting for port $port ... "
  while true; do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "READY"; break
    fi
    now=$(date +%s)
    if [ $((now - start)) -gt "$MAX_WAIT" ]; then
      echo "TIMEOUT after ${MAX_WAIT}s"
      docker compose --env-file "$ENV_FILE" -f "$EXP_ROOT/docker/docker-compose.yml" logs --tail 40 || true
      exit 1
    fi
    sleep $INTERVAL
  done
done

echo "=== All endpoints ready ==="
# Cosmetic only — must never fail the caller (pipefail + set -e safe).
set +e +o pipefail
for port in $PORTS; do
  m=$(curl -s "http://127.0.0.1:${port}/v1/models" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  echo "  :$port  ${m:-?}"
done
exit 0
