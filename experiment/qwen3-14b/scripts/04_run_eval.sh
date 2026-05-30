#!/usr/bin/env bash
# Full driver: for each model (base, v3.5, v3.1) bring up 4 vLLM endpoints on
# GPUs 4-7, run all 4 benchmarks × 3 protocols, then tear the model down before
# the next. One command runs the whole matrix; safe to nohup.
#
#   bash scripts/04_run_eval.sh                 # full run
#   LIMIT=10 bash scripts/04_run_eval.sh        # 10-question smoke test
#   MODELS="v35" BENCHES="tombench" bash scripts/04_run_eval.sh   # subset
#
# The eval client runs inside the serve image (which ships openai + tqdm) so the
# host needs no python deps.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="$EXP_ROOT/configs/deploy.env.181"
# shellcheck disable=SC1090
source "$ENV_FILE"

MODELS="${MODELS:-base v35 v31}"
BENCHES="${BENCHES:-tombench hitom socialiqa emobench}"
PROTOCOLS="${PROTOCOLS:-direct,direct_think,cot}"
CONC="${CONCURRENCY:-64}"
CLIENT_IMAGE="${CLIENT_IMAGE:-qwen3-tom-serve-eval-dp4:latest}"

LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

OUT_DIR="$EXP_ROOT/output"
LOG_DIR="$EXP_ROOT/logs"
CACHE_DIR="$OUT_DIR/cache"
mkdir -p "$LOG_DIR" "$CACHE_DIR"
for b in $BENCHES; do mkdir -p "$OUT_DIR/$b"; done

ENDPOINTS=""; for p in $PORTS; do ENDPOINTS="$ENDPOINTS 127.0.0.1:$p"; done

bench_data() {
  case "$1" in
    tombench)  echo "$EVAL_TOMBENCH" ;;
    hitom)     echo "$EVAL_HITOM" ;;
    socialiqa) echo "$EVAL_SOCIALIQA" ;;
    emobench)  echo "$EVAL_EMOBENCH" ;;
  esac
}
model_id()  { case "$1" in base) echo "$BASE_MODEL_ID";; v35) echo "$V35_MODEL_ID";; v31) echo "$V31_MODEL_ID";; esac; }
model_name(){ case "$1" in base) echo "$BASE_MODEL_NAME";; v35) echo "$V35_MODEL_NAME";; v31) echo "$V31_MODEL_NAME";; esac; }

# Run the python client inside the serve image (host network → reaches 127.0.0.1).
docker_py() {
  docker run --rm --network host \
    --user "$(id -u):$(id -g)" \
    -e PYTHONPATH="$EXP_ROOT/scripts" -e HOME=/tmp \
    -v "$REMOTE_BASE":"$REMOTE_BASE" \
    -w "$EXP_ROOT" \
    --entrypoint python3 \
    "$CLIENT_IMAGE" "$@"
}

echo "############################################################"
echo "# Qwen3-14B full eval"
echo "#   models:    $MODELS"
echo "#   benches:   $BENCHES"
echo "#   protocols: $PROTOCOLS   limit: ${LIMIT:-none}"
echo "#   started:   $(date -Iseconds)"
echo "############################################################"

for mk in $MODELS; do
  mid="$(model_id "$mk")"; mname="$(model_name "$mk")"
  echo ""
  echo "================= MODEL $mk ($mname) ================="
  bash "$HERE/01_serve_up.sh" "$mk"
  bash "$HERE/02_wait_ready.sh"

  for b in $BENCHES; do
    data="$(bench_data "$b")"
    out="$OUT_DIR/$b/${mid}.json"
    logf="$LOG_DIR/run_${mid}_${b}.log"
    echo "----- $mid / $b / $PROTOCOLS -----  (log: $logf)"
    # shellcheck disable=SC2086
    docker_py "$EXP_ROOT/scripts/parallel_eval.py" \
      --model "$mname" --model-id "$mid" \
      --endpoints $ENDPOINTS \
      --benchmark "$b" --data "$data" \
      --protocols "$PROTOCOLS" \
      --output "$out" \
      --cache-dir "$CACHE_DIR" \
      --concurrency "$CONC" \
      $LIMIT_FLAG 2>&1 | tee "$logf"
  done

  echo "----- tearing down $mk -----"
  bash "$HERE/06_serve_down.sh"
done

echo ""
echo "=== ALL DONE  $(date -Iseconds) ==="
ls -la "$OUT_DIR"/*/*.json 2>/dev/null || true
