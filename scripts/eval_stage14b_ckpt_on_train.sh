#!/usr/bin/env bash
# TRAIN-side eval pipeline for Stage 14b ckpts.
# Runs the eval client INSIDE the TRAIN docker (qwen3-tom-train image),
# talking to vLLM via localhost — avoids cross-host latency and follows
# the project rule "evals run on TRAIN".
#
# Prereq: ckpt-${CKPT} already converted to HF (use scripts/eval_stage14b_ckpt.sh
# first, OR convert manually).
#
# Usage: ./scripts/eval_stage14b_ckpt_on_train.sh <ckpt_step> [gpu_id]
#   <ckpt_step> ∈ {50, 100, 150, 199}
#   [gpu_id] default 1 (avoid GPU 0 if shared)
set -e
source configs/deploy.env

CKPT="${1:?usage: $0 <ckpt_step> [gpu_id]}"
GPU_ID="${2:-1}"
EXP_NAME="qwen3-14B-tombench-rlvr-stage14b-1x8"
SUBDIR="20260521-154242"
SRC_PATH="/mnt/output/${EXP_NAME}/${SUBDIR}/checkpoint-${CKPT}"
HF_DIR="${EXP_NAME}-hf-ckpt${CKPT}"

echo "Ckpt:    ${CKPT}"
echo "GPU:     ${GPU_ID}"

# Step 1: convert if not already done
echo
echo "=== Convert ckpt-${CKPT} (skip if HF dir exists) ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  if [ -f /data_nvme/grj-projects/tom-output/${HF_DIR}/config.json ]; then
    echo '  already converted, skipping'
  else
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
  fi
"

# Step 2: launch vLLM on TRAIN
echo
echo "=== vLLM serve ${HF_DIR} on GPU ${GPU_ID} ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  docker rm -f eval-serve-stage14b 2>/dev/null || true
  docker run -d --name eval-serve-stage14b \
    --gpus all -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
    --ipc host --shm-size 16gb -p 8000:8000 \
    -v /data_nvme/grj-projects/tom-output:/mnt/output \
    --entrypoint python qwen3-tom-train:latest \
    -m vllm.entrypoints.openai.api_server \
    --model /mnt/output/${HF_DIR} \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --served-model-name eval-target
"

# Wait for vLLM ready (poll from local — works regardless of where eval runs)
echo "Waiting for vLLM ready..."
for i in {1..60}; do
  if curl -fs http://172.16.120.181:8000/v1/models 2>/dev/null | grep -q eval-target; then
    echo "  vLLM ready after ${i}*5s"
    break
  fi
  sleep 5
done

# Step 3: rsync eval scripts + data + cache to TRAIN
echo
echo "=== Sync eval client to TRAIN ==="
rsync -av --quiet -e "ssh -i $TRAIN_SSH_KEY" \
  scripts/eval/ \
  h800@172.16.120.181:/data_nvme/grj-projects/qwen3-tom/scripts/eval/
rsync -av --quiet -e "ssh -i $TRAIN_SSH_KEY" \
  production_frozen/data/tombench_eval.jsonl \
  h800@172.16.120.181:/data_nvme/grj-projects/tom-data/
# (cache rsync optional — fresh ckpt won't reuse old cache anyway since cache is keyed by model name)

# Step 4: run eval ON TRAIN, talking to localhost vLLM
echo
echo "=== Run full 5718 eval on TRAIN (3 protocols) ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd /data_nvme/grj-projects/qwen3-tom && \
  mkdir -p logs output/eval output/eval_cache && \
  docker run --rm \
    --network host \
    -e OPENAI_API_KEY=dummy \
    -v /data_nvme/grj-projects/qwen3-tom:/workspace \
    -v /data_nvme/grj-projects/tom-data:/data_in \
    -w /workspace \
    --entrypoint python qwen3-tom-train:latest \
    scripts/eval/run_tombench.py \
      --backend openai \
      --base-url http://localhost:8000/v1 \
      --model eval-target \
      --data /data_in/tombench_eval.jsonl \
      --protocols direct,cot,del_tom \
      --concurrency 32 \
      --output /workspace/output/eval/stage14b_ckpt${CKPT}_full5718.json \
      2>&1 | tee logs/eval_stage14b_ckpt${CKPT}_full5718.log | tail -3
"

# Step 5: rsync result back
echo
echo "=== Pull result back ==="
rsync -av -e "ssh -i $TRAIN_SSH_KEY" \
  h800@172.16.120.181:/data_nvme/grj-projects/qwen3-tom/output/eval/stage14b_ckpt${CKPT}_full5718.json \
  output/eval/

# Step 6: stop vLLM
echo
echo "=== Stop vLLM ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" 'docker stop eval-serve-stage14b && docker rm eval-serve-stage14b' || true

echo
echo "=== Summary ==="
ls -la output/eval/stage14b_ckpt${CKPT}_full5718.json
