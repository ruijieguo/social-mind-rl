#!/usr/bin/env bash
# Track B: launch ExploreToM data generation on TRAIN host using stage 8 vLLM endpoint.
# This generates ~1500 program-guided ToM examples adversarial to stage 8.
#
# Prereq: vLLM serving stage 8 14B at http://localhost:8000 on TRAIN host
# Output: framework/ExploreToM/logs/*.jsonl (then rsync back to data/tom/raw/)
set -e
source configs/deploy.env

echo "=== Phase 1: generate story contexts (~5 min) ==="
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
  cd $TRAIN_PATH/framework/ExploreToM && \
  docker run --rm \
    --network host \
    -v $TRAIN_PATH/framework/ExploreToM:/work \
    -w /work \
    --entrypoint python qwen3-tom-train:latest \
    story_context_generator.py \
      --num_elements_by_class 6 \
      --num_contexts_to_generate 200 \
      --model_name qwen3-14b-tom-stage8 \
      --model_access_method vllm-api 2>&1 | tail -30
"

echo "=== Phase 2: A* search (~30 min, 4 shards) ==="
for i in 0 1 2 3; do
  ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" "
    cd $TRAIN_PATH/framework/ExploreToM && \
    nohup docker run --rm \
      --network host \
      -v $TRAIN_PATH/framework/ExploreToM:/work \
      -w /work \
      --entrypoint python qwen3-tom-train:latest \
      story_structure_searcher.py \
        --experiment_to_run search \
        --model_name qwen3-14b-tom-stage8 \
        --model_access_method vllm-api \
        --a_star_neighbor_priority weight-goal4 \
        --i $i > logs/search_$i.log 2>&1 &
  "
done
echo "shard 0-3 launched, will run in parallel"
echo "Check: ssh \$TRAIN_HOST 'ls $TRAIN_PATH/framework/ExploreToM/logs/'"
echo
echo "After all shards complete (~30-60 min), run Phase 3 manually:"
echo "  for i in 0 1 2 3; do"
echo "    ssh \$TRAIN_HOST 'cd \$TP/framework/ExploreToM && python story_structure_infiller.py --i \$i &'"
echo "  done"
