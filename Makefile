# Qwen3-8B ToMBench RLVR project Makefile

SHELL := /usr/bin/env bash
-include configs/deploy.env
export

TRAIN_HOST_HOSTONLY = $(shell echo "$(TRAIN_HOST)" | sed 's/.*@//')

.PHONY: help env-check build-data baseline sync-up sync-down \
        train-stage1 train-stage2 train-stage3-l3 \
        serve-launch serve-stop serve-url eval-final analyze \
        pipeline-stage1 pipeline-stage2 pipeline-l3 \
        test-reward test-eval test-data

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "%-22s %s\n", $$1, $$2}'

# ============================================================
# DEV local (macOS) targets
# ============================================================

env-check: ## Phase 0: verify DEV/TRAIN environment
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  python scripts/env_check.py

build-data: ## Phase 1: build training data + ToMBench eval set
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  python scripts/data/merge_and_dedupe.py

baseline: ## Phase 2: evaluate Qwen3-8B (nt+t) and deepseek-v4-pro on ToMBench
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  python scripts/eval/run_tombench.py --preset baseline-all

test-reward: ## Run reward worker unit tests
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  pytest framework/ROLL/tests/test_tom_mcq_reward.py -v

test-eval: ## Run eval framework unit tests
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  pytest scripts/eval/tests/ -v

test-data: ## Run data construction unit tests
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  pytest scripts/data/tests/ -v

# ============================================================
# Sync to/from TRAIN
# ============================================================

sync-up: ## rsync code + data → TRAIN
	bash scripts/deploy/sync_to_train.sh

sync-down: ## rsync ckpt + logs ← TRAIN
	bash scripts/deploy/sync_from_train.sh

# ============================================================
# TRAIN training (triggered from DEV via SSH)
# ============================================================

train-stage1: ## Phase 4: stage-1 RL training (4k × 200 steps) on TRAIN
	ssh -i $(TRAIN_SSH_KEY) $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/train/docker-compose.yml \
	  --env-file configs/deploy.env \
	  run --rm -e STAGE=stage1 train"

train-stage2: ## Phase 5: stage-2 RL training (8k × 500 steps) on TRAIN
	ssh -i $(TRAIN_SSH_KEY) $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/train/docker-compose.yml \
	  --env-file configs/deploy.env \
	  run --rm -e STAGE=stage2 train"

train-stage3-l3: ## Phase 9: L3 fallback (process-reward) on TRAIN
	ssh -i $(TRAIN_SSH_KEY) $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/train/docker-compose.yml \
	  --env-file configs/deploy.env \
	  run --rm -e STAGE=stage3_l3 train"

# ============================================================
# Serve trained model on TRAIN
# ============================================================

serve-launch: ## Launch vLLM serve container on TRAIN
	ssh -i $(TRAIN_SSH_KEY) $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/serve/docker-compose.yml up -d serve"

serve-stop: ## Stop vLLM serve container on TRAIN
	ssh -i $(TRAIN_SSH_KEY) $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/serve/docker-compose.yml down"

serve-url: ## Print the OpenAI-compatible base_url to use for eval
	@echo "Base URL: http://$(TRAIN_HOST_HOSTONLY):$(SERVE_PORT)/v1"
	@echo "Model id: qwen3-8b-tom"
	@echo
	@echo "Test:"
	@echo "  curl http://$(TRAIN_HOST_HOSTONLY):$(SERVE_PORT)/v1/models"

# ============================================================
# DEV evaluation of trained model
# ============================================================

eval-final: ## Phase 6: evaluate trained model (3 protocols × ZH+EN)
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  python scripts/eval/run_tombench.py \
	    --backend openai \
	    --base-url http://$(TRAIN_HOST):$(SERVE_PORT)/v1 \
	    --model qwen3-8b-tom \
	    --protocols direct,cot,del_tom \
	    --output output/eval/final.json

# ============================================================
# Analysis (DEV)
# ============================================================

analyze: ## Generate training curves, eval diff, error audit
	docker compose -f docker/dev/docker-compose.yml run --rm dev bash -c "\
	  python scripts/analysis/plot_training_curves.py && \
	  python scripts/analysis/diff_eval_results.py && \
	  python scripts/analysis/error_audit.py"

# ============================================================
# Pipelines (composition)
# ============================================================

pipeline-stage1: build-data baseline sync-up train-stage1 sync-down analyze
pipeline-stage2: sync-up train-stage2 sync-down serve-launch eval-final analyze
pipeline-l3:     sync-up train-stage3-l3 sync-down serve-launch eval-final analyze
