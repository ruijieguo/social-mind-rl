#!/usr/bin/env bash
# Eval a Stage 11D / Stage 12 checkpoint after training completes.
# Steps:
#   1. SSH to TRAIN: convert Megatron ckpt -> HF format
#   2. Launch vLLM serve on stage X HF model (port 8000)
#   3. Run full 5718 eval (direct + cot + del_tom)
#   4. Run subset500 + clean 4551 eval
#   5. Stop vLLM
#
# Usage: ./scripts/eval_stage_post_train.sh <stage_name>
#   stage_name = stage11d_continue_1x8_14b | stage12_1x8_14b

set -e
source configs/deploy.env

STAGE="${1:?usage: $0 <stage_name>}"
EXP_NAME=$(grep '^exp_name:' configs/tombench-rlvr/rlvr_config_${STAGE}.yaml | sed 's/.*"\(.*\)"/\1/')
HF_DIR="${EXP_NAME}-hf-final"
echo "Stage: $STAGE"
echo "Exp:   $EXP_NAME"
echo "HF:    $HF_DIR"

# Step 1: convert Megatron -> HF on TRAIN using mcore_adapter inside the train container
echo
echo "=== Convert ckpt to HF format on TRAIN ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd $TRAIN_PATH && \
  docker compose -f docker/train/docker-compose.yml \
    --env-file configs/deploy.env \
    run --rm --build \
    --entrypoint python \
    train \
    scripts/deploy/convert_megatron_to_hf.py \
      --src /mnt/output/${EXP_NAME}/final \
      --dst /mnt/output/${HF_DIR} \
      --base-model Qwen/Qwen3-14B
"

# Step 2: launch vLLM serve
echo
echo "=== vLLM serve $HF_DIR on TRAIN ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  docker stop eval-serve 2>/dev/null || true
  docker rm eval-serve 2>/dev/null || true
  docker run -d --name eval-serve \
    --gpus device=0 --ipc host --shm-size 16gb -p 8000:8000 \
    -v $TRAIN_OUTPUT_DIR:/mnt/output \
    --entrypoint python qwen3-tom-train:latest \
    -m vllm.entrypoints.openai.api_server \
    --model /mnt/output/$HF_DIR \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --served-model-name eval-target
  sleep 60
"

# Step 3: full 5718 eval (3 protocols)
echo
echo "=== Full 5718 eval (3 protocols) ==="
LOG_F="logs/eval_${STAGE}_full5718_$(date +%Y%m%d_%H%M%S).log"
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy \
  -v /Users/jaredguo-mini/develop/training:/workspace \
  -w /workspace dev \
  python scripts/eval/run_tombench.py \
    --backend openai \
    --base-url http://$TRAIN_HOST_IP:8000/v1 \
    --model eval-target \
    --data /workspace/production_frozen/data/tombench_eval.jsonl \
    --protocols direct,cot,del_tom \
    --concurrency 32 \
    --output /workspace/output/eval/${STAGE}_full5718.json \
    > "$LOG_F" 2>&1 || echo "  eval failed, check $LOG_F"

# Step 4: subset500 + clean 4551
echo
echo "=== Subset500 + clean 4551 eval ==="
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy \
  -v /Users/jaredguo-mini/develop/training:/workspace \
  -w /workspace dev \
  python scripts/eval/run_tombench.py \
    --backend openai \
    --base-url http://$TRAIN_HOST_IP:8000/v1 \
    --model eval-target \
    --data /workspace/production_frozen/data/tombench_eval_clean.jsonl \
    --protocols direct,cot,del_tom \
    --concurrency 32 \
    --output /workspace/output/eval/${STAGE}_clean4551.json

# Step 5: stop vLLM
echo
echo "=== Stop vLLM ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" 'docker stop eval-serve && docker rm eval-serve'

echo
echo "=== Summary ==="
python3 -c "
import json
for p in ['${STAGE}_full5718', '${STAGE}_clean4551']:
    try:
        recs = json.load(open(f'output/eval/{p}.json'))
        from collections import Counter
        c = Counter(r['protocol'] for r in recs)
        for proto in ['direct','cot','del_tom']:
            sub = [r for r in recs if r['protocol']==proto]
            if sub:
                acc = sum(r['correct'] for r in sub)/len(sub)
                print(f'  {p:<35} {proto:<8} {acc:.4f}')
    except Exception as e:
        print(f'  {p}: error: {e}')
"
