#!/usr/bin/env bash
# Stage 14b ckpt-specific eval: convert one ckpt to HF, serve on GPU 1 (GPU 0 occupied),
# then run full 5718 eval (direct + cot + del_tom).
#
# Usage: ./scripts/eval_stage14b_ckpt.sh <ckpt_step>
#   <ckpt_step> ∈ {50, 100, 150, 199}
set -e
source configs/deploy.env

CKPT="${1:?usage: $0 <ckpt_step>}"
EXP_NAME="qwen3-14B-tombench-rlvr-stage14b-1x8"
SUBDIR="20260521-154242"
SRC_PATH="/mnt/output/${EXP_NAME}/${SUBDIR}/checkpoint-${CKPT}"
HF_DIR="${EXP_NAME}-hf-ckpt${CKPT}"
GPU_ID="${GPU_ID:-1}"  # default GPU 1 (avoid GPU 0 used by other experiment)

echo "Ckpt:    ${CKPT}"
echo "Source:  ${SRC_PATH}"
echo "HF dst:  /mnt/output/${HF_DIR}"
echo "GPU:     ${GPU_ID}"

# Step 1: Megatron -> HF on TRAIN (uses GPU briefly during conversion)
echo
echo "=== Convert ckpt-${CKPT} Megatron → HF ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd $TRAIN_PATH && \
  docker compose -f docker/train/docker-compose.yml \
    --env-file configs/deploy.env \
    run --rm --build \
    --entrypoint python \
    train \
    scripts/deploy/convert_megatron_to_hf.py \
      --src ${SRC_PATH} \
      --dst /mnt/output/${HF_DIR} \
      --base-model Qwen/Qwen3-14B
"

# Step 2: launch vLLM on GPU ${GPU_ID}
echo
echo "=== vLLM serve ${HF_DIR} on GPU ${GPU_ID} ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  docker rm -f eval-serve-stage14b 2>/dev/null || true
  docker run -d --name eval-serve-stage14b \
    --gpus all -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
    --ipc host --shm-size 16gb -p 8000:8000 \
    -v $TRAIN_OUTPUT_DIR:/mnt/output \
    --entrypoint python qwen3-tom-train:latest \
    -m vllm.entrypoints.openai.api_server \
    --model /mnt/output/${HF_DIR} \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --served-model-name eval-target
"

# Wait for vLLM ready
echo "Waiting for vLLM ready..."
for i in {1..60}; do
  if curl -fs http://${TRAIN_HOST_IP}:8000/v1/models 2>/dev/null | grep -q eval-target; then
    echo "  vLLM ready after ${i}*5s"
    break
  fi
  sleep 5
done

# Step 3: full 5718 eval (3 protocols)
echo
echo "=== Full 5718 eval ==="
LOG_F="logs/eval_stage14b_ckpt${CKPT}_full5718_$(date +%Y%m%d_%H%M%S).log"
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy \
  -v /Users/jaredguo-mini/develop/training:/workspace \
  -w /workspace dev \
  python scripts/eval/run_tombench.py \
    --backend openai \
    --base-url http://${TRAIN_HOST_IP}:8000/v1 \
    --model eval-target \
    --data /workspace/production_frozen/data/tombench_eval.jsonl \
    --protocols direct,cot,del_tom \
    --concurrency 32 \
    --output /workspace/output/eval/stage14b_ckpt${CKPT}_full5718.json \
    > "$LOG_F" 2>&1
echo "  log: $LOG_F"

# Step 4: stop vLLM
echo
echo "=== Stop vLLM ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" 'docker stop eval-serve-stage14b && docker rm eval-serve-stage14b' || true

echo
echo "=== Summary ==="
ls -la output/eval/stage14b_ckpt${CKPT}_full5718.* 2>&1 | head
