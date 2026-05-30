#!/usr/bin/env bash
# Post-training auto-eval for Stage 22 (Plan A) — runs detached ON .191.
# Waits for training to finish (GPUs free), converts a strided set of checkpoints
# Megatron→HF, evaluates each + base on the 4 benchmarks × 3 protocols (same
# harness as the full-eval), then aggregates the 4-bench best-protocol mean and
# compares against base (measured here) and v3.1 (cited 0.7305 from prior eval).
#
# Verdict to look for: a Stage-22 ckpt with 4-bench mean > base 0.7603 AND
# Hi-ToM cot not collapsed (≥0.76) → Plan A worked (gained without the tax).
set -uo pipefail

REPO=/home/h800/grj-projects/qwen3-tom
EXP=$REPO/experiment/qwen3-14b
OUTROOT=/home/h800/grj-projects/tom-output
MODELS=/home/h800/grj-projects/models
RUN_GLOB="$OUTROOT/qwen3-14B-tombench-rlvr-stage22-planA-1x8"
HFROOT=$OUTROOT/stage22-hf            # converted HF ckpts land here
LOG=$EXP/logs/posteval.log
mkdir -p "$EXP/logs" "$HFROOT" "$EXP/output"/{tombench,emobench,socialiqa,hitom}
exec > >(tee -a "$LOG") 2>&1

EVAL_CKPTS="${EVAL_CKPTS:-50 100 150 200}"   # strided: catch peak + collapse
PROTOCOLS="${PROTOCOLS:-direct,direct_think,cot}"
CONC="${CONCURRENCY:-64}"
PORTS="8001 8002 8003 8004"
ENDPOINTS=""; for p in $PORTS; do ENDPOINTS="$ENDPOINTS 127.0.0.1:$p"; done
SERVE=$EXP/docker/serve_dp4.yml
TRAIN_IMG=qwen3-tom-train:latest
SERVE_IMG=qwen3-tom-serve-eval-dp4:latest
declare -A DATA=(
  [tombench]=$REPO/data/tom/tombench_eval.jsonl
  [hitom]=$REPO/data/eval/hitom_eval.jsonl
  [socialiqa]=$REPO/data/eval/socialiqa_eval.jsonl
  [emobench]=$REPO/data/eval/emobench_eval.jsonl
)

echo "[posteval] $(date -Iseconds) waiting for training to finish..."
for i in $(seq 1 96); do  # up to ~19h
  up=$(docker ps --filter name=train-train-1 --format '{{.Status}}')
  [ -z "$up" ] && { echo "[posteval] training ended"; break; }
  sleep 600
done
sleep 30  # let GPUs settle

RUNDIR=$(ls -d $RUN_GLOB/*/ 2>/dev/null | tail -1)
echo "[posteval] run dir: $RUNDIR"

serve_up(){ MODEL_HOST_DIR=/home/h800/grj-projects MODEL_PATH="$1" SERVED_NAME="$2" \
    GPU_UTIL=0.85 MAX_MODEL_LEN=16384 SERVE_PORT_BASE=8001 \
    docker compose -f "$SERVE" up -d; }
serve_down(){ docker compose -f "$SERVE" down 2>/dev/null; }
wait_ready(){ for k in $(seq 1 90); do ok=0; for p in $PORTS; do
    curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && ok=$((ok+1)); done
    [ "$ok" -eq 4 ] && { echo "[posteval] 4 endpoints ready"; return 0; }; sleep 12; done; return 1; }
docker_py(){ docker run --rm --network host --user "$(id -u):$(id -g)" \
    -e PYTHONPATH=$EXP/scripts -e HOME=/tmp -v "$REPO":"$REPO" -w "$EXP" \
    --entrypoint python3 "$SERVE_IMG" "$@"; }

eval_model(){  # $1=model_path $2=model_id $3=served_name
  local mp="$1" mid="$2" sn="$3"
  echo "[posteval] === serve+eval $mid ($mp) ==="
  serve_down; serve_up "$mp" "$sn"; wait_ready || { echo "[posteval] $mid serve FAILED"; serve_down; return 1; }
  for b in tombench hitom socialiqa emobench; do
    echo "[posteval] $mid / $b"
    docker_py "$EXP/scripts/parallel_eval.py" --model "$sn" --model-id "$mid" \
      --endpoints $ENDPOINTS --benchmark "$b" --data "${DATA[$b]}" \
      --protocols "$PROTOCOLS" --output "$EXP/output/$b/$mid.json" \
      --cache-dir "$EXP/output/cache" --concurrency "$CONC" 2>&1 | tail -3
  done
  serve_down
}

# 1) base column (measured on this exact .191 harness)
eval_model "$MODELS/Qwen3-14B" base qwen3-14b-base

# 2) each selected ckpt: convert → eval
for n in $EVAL_CKPTS; do
  src="$RUNDIR/checkpoint-$n"
  [ -d "$src" ] || { echo "[posteval] ckpt-$n missing, skip"; continue; }
  dst="$HFROOT/ckpt-$n"
  if [ ! -f "$dst/config.json" ]; then
    echo "[posteval] convert ckpt-$n → HF"
    docker run --rm --gpus all --network host \
      -e PYTHONPATH=/workspace:/workspace/framework/ROLL -e HF_HOME=$MODELS/.cache/huggingface \
      -e USE_MODELSCOPE=1 -v "$REPO":/workspace -v /home/h800/grj-projects:/home/h800/grj-projects \
      -w /workspace --entrypoint python "$TRAIN_IMG" \
      framework/ROLL/mcore_adapter/tools/convert.py \
        --checkpoint_path "$src" --output_path "$dst" --bf16 2>&1 | tail -4
  fi
  [ -f "$dst/config.json" ] && eval_model "$dst" "ckpt$n" "qwen3-14b-stage22-ckpt$n" \
    || echo "[posteval] convert ckpt-$n FAILED, skip eval"
done

# 3) aggregate
echo "[posteval] === AGGREGATE ==="
docker_py "$EXP/scripts/posteval_summary.py" --results-dir "$EXP/output" --ckpts "$EVAL_CKPTS" || true
echo "[posteval] POSTEVAL_DONE $(date -Iseconds)"
