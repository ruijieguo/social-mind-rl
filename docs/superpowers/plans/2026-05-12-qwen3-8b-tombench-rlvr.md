# Qwen3-8B ToMBench RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Post-train Qwen/Qwen3-8B with GRPO on ROLL to maximize ToMBench accuracy, approaching or surpassing deepseek-v4-pro, delivered as a Docker-based DEV↔TRAIN workflow.

**Architecture:** DEV (macOS) builds data + evaluates via APIs; TRAIN (16×H800) runs ROLL RLVR pipeline in Docker; rsync over SSH for code/data/ckpt sync; OpenAI-compatible vLLM serves the trained model back to DEV for evaluation.

**Tech Stack:** Python 3.10, Docker, ROLL framework (Megatron + vLLM), Qwen3-8B, DashScope/DeepSeek APIs, GRPO + Clip-Higher + Dynamic Sampling, rule-based reward (R_fmt × R_out × R_len).

**Spec:** `docs/superpowers/specs/2026-05-11-qwen3-8b-tombench-rlvr-design.md`

---

## File Structure

Files this plan creates, grouped by responsibility:

**Repo bootstrap**
- `Makefile` — top-level commands (`make build-data`, `make pipeline-stage2`, ...)
- `configs/deploy.env.example` — deployment env vars template
- `docs/README.md`, `docs/runbook.md`, `docs/data-card.md`, `docs/eval-protocol.md`

**Docker images**
- `docker/dev/{Dockerfile, docker-compose.yml}`
- `docker/train/{Dockerfile, docker-compose.yml, entrypoint.sh}`
- `docker/serve/{Dockerfile, docker-compose.yml, entrypoint.sh}`
- `docker/.dockerignore`

**Data construction (DEV)**
- `scripts/data/build_tombench_eval.py` — download + transform ToMBench
- `scripts/data/build_hitom.py` — Hi-ToM via ToM-RL scripts
- `scripts/data/build_exploretom.py` — ExploreToM from HuggingFace
- `scripts/data/build_simpletom.py` — SimpleToM from HuggingFace
- `scripts/data/build_socialiqa.py` — SocialIQa from HuggingFace
- `scripts/data/synth_tomtype.py` — deepseek synthesis
- `scripts/data/merge_and_dedupe.py` — MinHash dedupe + split
- `scripts/data/schema.py` — shared data record dataclass
- `scripts/data/tests/test_*.py` — unit tests

**Evaluation framework (DEV)**
- `scripts/eval/clients.py` — OpenAI-compatible client (dashscope/deepseek/openai/local)
- `scripts/eval/extractors.py` — answer extractor (direct/cot/del_tom)
- `scripts/eval/run_tombench.py` — main runner
- `scripts/eval/report.py` — markdown/json aggregation
- `scripts/eval/tests/test_*.py` — unit tests

**ROLL reward worker**
- `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` — L2 reward
- `framework/ROLL/tests/test_tom_mcq_reward.py` — 6 unit tests

**ROLL configs**
- `configs/tombench-rlvr/rlvr_config_stage1.yaml`
- `configs/tombench-rlvr/rlvr_config_stage2.yaml`
- `configs/tombench-rlvr/rlvr_config_stage3_l3.yaml`

**Deploy scripts**
- `scripts/deploy/sync_to_train.sh` / `sync_from_train.sh`
- `scripts/deploy/train_launch.sh` / `serve_launch.sh`
- `scripts/deploy/train_monitor.py` — early-stop guard
- `scripts/deploy/track_best_ckpt.py` — best-ckpt symlink
- `scripts/deploy/convert_megatron_to_hf.py`

**Analysis (DEV)**
- `scripts/analysis/plot_training_curves.py`
- `scripts/analysis/diff_eval_results.py`
- `scripts/analysis/error_audit.py`

**Misc**
- `scripts/env_check.py` — phase 0 environment check
- `.gitignore` (already exists, will extend)

---

## Phase 0 — Repo bootstrap

### Task 0.1: Extend .gitignore

**Files:**
- Modify: `.gitignore` (already exists)

- [ ] **Step 1: Add additional ignore patterns**

Open `.gitignore` and append:

```
# Eval cache
output/eval_cache/

# TensorBoard
output/tensorboard/

# Checkpoints (synced from TRAIN)
output/checkpoints/

# API usage logs
output/api_usage.jsonl

# manifests (regenerated)
output/manifests/

# Local docker volumes
docker_volumes/
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jaredguo-mini/develop/training
git add .gitignore
git commit -m "chore: extend .gitignore for runtime artifacts"
```

---

### Task 0.2: Create configs/deploy.env.example

**Files:**
- Create: `configs/deploy.env.example`
- Create: `configs/.gitkeep`

- [ ] **Step 1: Write deploy.env.example**

```bash
# TRAIN server SSH access
TRAIN_HOST=user@training-server.example.com
TRAIN_SSH_KEY=~/.ssh/id_rsa_train
TRAIN_PATH=/data/cpfs_0/projects/qwen3-tom

# TRAIN paths (inside the training server)
TRAIN_DATA_DIR=/data/cpfs_0/tom-data
TRAIN_MODELS_DIR=/data/cpfs_0/models
TRAIN_OUTPUT_DIR=/data/cpfs_0/tom-output

# vLLM serve endpoint (TRAIN_HOST + port)
SERVE_PORT=8000

# DEV local paths (relative to repo root)
DEV_DATA_DIR=./data
DEV_OUTPUT_DIR=./output

# NOTE: API keys are NOT stored here. Export them in your shell:
#   export DEEPSEEK_API_KEY=...
#   export DASHSCOPE_API_KEY=...
```

- [ ] **Step 2: Add to git and commit**

```bash
cd /Users/jaredguo-mini/develop/training
mkdir -p configs
touch configs/.gitkeep
git add configs/
git commit -m "chore: add deploy.env.example template"
```

---

### Task 0.3: Initialize doc skeleton

**Files:**
- Create: `docs/README.md`
- Create: `docs/runbook.md`
- Create: `docs/data-card.md`
- Create: `docs/eval-protocol.md`

- [ ] **Step 1: Write docs/README.md**

```markdown
# Qwen3-8B ToMBench RLVR

Post-train Qwen/Qwen3-8B with GRPO on ROLL to maximize ToMBench accuracy.

**Design spec:** `docs/superpowers/specs/2026-05-11-qwen3-8b-tombench-rlvr-design.md`

## Quick start

```bash
cp configs/deploy.env.example configs/deploy.env
# edit configs/deploy.env with your TRAIN host info

export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...

make env-check
make build-data
make baseline
make pipeline-stage1
make pipeline-stage2
```

## Documents
- `docs/runbook.md` — operational runbook for each stage
- `docs/data-card.md` — training data sources, licenses, dedupe audit
- `docs/eval-protocol.md` — three evaluation protocols defined precisely
```

- [ ] **Step 2: Write skeleton for runbook / data-card / eval-protocol**

Each file gets a placeholder skeleton; will be filled in later tasks.

`docs/runbook.md`:
```markdown
# Runbook

Operational steps for each stage of the project. See `docs/superpowers/specs/...` for design rationale.

## Stage 0 — Environment check
[to be filled by Task 11.2]

## Stage 1 — Data construction
[to be filled by Task 11.2]

## Stage 2 — Baseline measurement
[to be filled by Task 11.2]
```

`docs/data-card.md`:
```markdown
# Data Card

Per-dataset card for ToM training data. Will be auto-generated by Task 4.6 after dedupe completes.
```

`docs/eval-protocol.md`:
```markdown
# Evaluation Protocols

Three protocols are defined. [Filled in Task 11.4]
```

- [ ] **Step 3: Commit**

```bash
cd /Users/jaredguo-mini/develop/training
git add docs/
git commit -m "docs: add README + runbook/data-card/eval-protocol skeletons"
```

---

### Task 0.4: Skeleton Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write Makefile**

```makefile
# Qwen3-8B ToMBench RLVR project Makefile

SHELL := /usr/bin/env bash
include configs/deploy.env
export

.PHONY: help env-check build-data baseline sync-up sync-down \
        train-stage1 train-stage2 train-stage3-l3 \
        serve-launch serve-stop eval-final analyze \
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
```

- [ ] **Step 2: Verify Makefile parses**

```bash
cd /Users/jaredguo-mini/develop/training
make help
```

Expected: prints the help table with all targets.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add top-level Makefile with all stage targets"
```

---

## Phase 1 — DEV Docker image

### Task 1.1: Write docker/dev/Dockerfile

**Files:**
- Create: `docker/dev/Dockerfile`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# DEV image — used on macOS for data construction, synthesis, eval, analysis.
# No CUDA. Lean Python 3.10 base.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/dev/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Mount points (mounted at runtime via compose)
VOLUME ["/workspace/scripts", "/workspace/data", "/workspace/output", "/workspace/configs"]

CMD ["bash"]
```

- [ ] **Step 2: Write docker/dev/requirements.txt**

```
openai>=1.40.0
datasets>=2.20.0
pyarrow>=15.0.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
tensorboard>=2.16.0
tqdm>=4.66.0
datasketch>=1.6.4
pytest>=8.0.0
jsonlines>=4.0.0
tenacity>=8.2.0
pyyaml>=6.0.1
huggingface-hub>=0.24.0
transformers>=4.45.0
```

- [ ] **Step 3: Commit**

```bash
mkdir -p docker/dev
# (write files above)
git add docker/dev/Dockerfile docker/dev/requirements.txt
git commit -m "build(docker): add DEV image (python 3.10, no CUDA)"
```

---

### Task 1.2: Write docker/dev/docker-compose.yml

**Files:**
- Create: `docker/dev/docker-compose.yml`
- Create: `docker/.dockerignore`

- [ ] **Step 1: Write docker-compose.yml**

```yaml
services:
  dev:
    build:
      context: ../..
      dockerfile: docker/dev/Dockerfile
    image: qwen3-tom-dev:latest
    working_dir: /workspace
    volumes:
      - ../../scripts:/workspace/scripts
      - ../../data:/workspace/data
      - ../../output:/workspace/output
      - ../../configs:/workspace/configs
      - ../../framework:/workspace/framework
      - ../../docs:/workspace/docs
    environment:
      DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:-}
      DASHSCOPE_API_KEY: ${DASHSCOPE_API_KEY:-}
      PYTHONPATH: /workspace
```

- [ ] **Step 2: Write docker/.dockerignore**

```
.git
.venv
venv
__pycache__
*.pyc
data
output
docker_volumes
*.swp
.DS_Store
node_modules
```

- [ ] **Step 3: Build the dev image to verify**

```bash
cd /Users/jaredguo-mini/develop/training
docker compose -f docker/dev/docker-compose.yml build dev
```

Expected: image builds in ~2-5 minutes, ends with `naming to docker.io/library/qwen3-tom-dev:latest`.

- [ ] **Step 4: Smoke-test the image**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev python -c "import openai, datasets, datasketch; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add docker/dev/docker-compose.yml docker/.dockerignore
git commit -m "build(docker): add DEV compose + .dockerignore; verified build"
```

---

### Task 1.3: Write scripts/env_check.py

**Files:**
- Create: `scripts/env_check.py`

- [ ] **Step 1: Write env_check.py**

```python
"""Phase 0 environment check.

Verifies:
- DEEPSEEK_API_KEY and DASHSCOPE_API_KEY are set
- Both API endpoints reachable with a minimal echo call
- Local mount points exist (data/, output/, configs/)
"""
import os
import sys
from pathlib import Path


def check_env_vars() -> list[str]:
    issues = []
    for var in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        if not os.environ.get(var):
            issues.append(f"missing env var: {var}")
    return issues


def check_mount_points() -> list[str]:
    issues = []
    for path in ("data", "output", "configs"):
        p = Path("/workspace") / path
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                issues.append(f"cannot create {p}: {e}")
    return issues


def check_deepseek() -> list[str]:
    from openai import OpenAI
    issues = []
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ["DEEPSEEK_API_KEY empty, skipping deepseek check"]
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=4,
        )
        content = resp.choices[0].message.content or ""
        print(f"  deepseek-v4-pro reachable; sample reply: {content!r}")
    except Exception as e:
        issues.append(f"deepseek api error: {e}")
    return issues


def check_dashscope() -> list[str]:
    from openai import OpenAI
    issues = []
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return ["DASHSCOPE_API_KEY empty, skipping dashscope check"]
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        resp = client.chat.completions.create(
            model="qwen3-8b",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=4,
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or ""
        print(f"  qwen3-8b (non-thinking) reachable; sample reply: {content!r}")
    except Exception as e:
        issues.append(f"dashscope api error: {e}")
    return issues


def main() -> int:
    print("=== Phase 0 environment check ===")
    all_issues: list[str] = []
    for name, fn in [
        ("env vars", check_env_vars),
        ("mount points", check_mount_points),
        ("deepseek api", check_deepseek),
        ("dashscope api", check_dashscope),
    ]:
        print(f"checking {name}...")
        issues = fn()
        for i in issues:
            print(f"  ! {i}")
        all_issues.extend(issues)

    if all_issues:
        print(f"\nFAILED with {len(all_issues)} issue(s)")
        return 1
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run env check**

```bash
cd /Users/jaredguo-mini/develop/training
export DEEPSEEK_API_KEY=<your_key>
export DASHSCOPE_API_KEY=<your_key>
make env-check
```

Expected: `ALL OK`. If any API key is missing or wrong, stop and fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add scripts/env_check.py
git commit -m "feat(env): add phase-0 environment check script"
```

---

## Phase 2 — Evaluation framework (TDD)

### Task 2.1: TDD answer extractors

**Files:**
- Create: `scripts/eval/__init__.py` (empty)
- Create: `scripts/eval/extractors.py`
- Create: `scripts/eval/tests/__init__.py` (empty)
- Create: `scripts/eval/tests/test_extractors.py`

- [ ] **Step 1: Write failing tests for direct extractor**

`scripts/eval/tests/test_extractors.py`:

```python
import pytest
from scripts.eval.extractors import extract_direct, extract_cot, vote_del_tom


# ---- Direct protocol: \boxed{X} first match ----

def test_direct_simple_boxed():
    assert extract_direct(r"\boxed{A}") == "A"


def test_direct_with_whitespace():
    assert extract_direct(r"  \boxed{B}  ") == "B"


def test_direct_picks_first_when_multiple():
    assert extract_direct(r"\boxed{A} then \boxed{C}") == "A"


def test_direct_fallback_first_capital_letter():
    # No boxed, fall back to first standalone capital letter A-D
    assert extract_direct("The answer is C.") == "C"


def test_direct_returns_none_when_nothing():
    assert extract_direct("blah blah") is None


def test_direct_invalid_letter_in_box_fallsback():
    # \boxed{Z} is invalid; should fall back to letter search
    assert extract_direct(r"\boxed{Z} but actually A") == "A"


# ---- CoT protocol: \boxed{X} last match ----

def test_cot_picks_last_boxed():
    text = "I think A first.\nWait, actually \\boxed{D}."
    assert extract_cot(text) == "D"


def test_cot_picks_last_when_multiple():
    assert extract_cot(r"\boxed{A} ... \boxed{B} ... \boxed{C}") == "C"


def test_cot_fallback_last_capital_letter_in_tail():
    text = "Long reasoning... final answer: B"
    assert extract_cot(text) == "B"


def test_cot_returns_none_when_nothing():
    assert extract_cot("just blabber") is None


# ---- DEL-ToM voting ----

def test_del_tom_majority_vote():
    answers = ["A", "A", "B", "A", "C", "A", "B", "A"]
    assert vote_del_tom(answers) == "A"


def test_del_tom_tie_breaks_alphabetically():
    answers = ["A", "B", "A", "B"]
    # Tie between A and B; alphabetic
    assert vote_del_tom(answers) == "A"


def test_del_tom_ignores_none():
    answers = ["A", None, "B", "A", None]
    assert vote_del_tom(answers) == "A"


def test_del_tom_all_none_returns_none():
    assert vote_del_tom([None, None, None]) is None
```

- [ ] **Step 2: Run tests, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_extractors.py -v
```

Expected: ImportError because `scripts/eval/extractors.py` doesn't exist.

- [ ] **Step 3: Implement extractors.py**

```python
"""Answer extractors for the three evaluation protocols."""
import re
from collections import Counter
from typing import Optional, Sequence

_BOXED_PATTERN = re.compile(r"\\boxed\{([A-D])\}")
_VALID = {"A", "B", "C", "D"}


def _first_capital_letter(text: str) -> Optional[str]:
    """Return the first standalone A/B/C/D in text, else None."""
    for ch in text:
        if ch in _VALID:
            return ch
    return None


def _last_capital_letter(text: str, tail_chars: int = 200) -> Optional[str]:
    """Return the last A/B/C/D within the last `tail_chars` of text."""
    tail = text[-tail_chars:]
    last = None
    for ch in tail:
        if ch in _VALID:
            last = ch
    return last


def extract_direct(text: str) -> Optional[str]:
    """Protocol 1: first \\boxed{X}; fallback to first capital letter A-D."""
    if not text:
        return None
    m = _BOXED_PATTERN.search(text)
    if m:
        return m.group(1)
    return _first_capital_letter(text)


def extract_cot(text: str) -> Optional[str]:
    """Protocol 2: last \\boxed{X}; fallback to last capital letter A-D in tail."""
    if not text:
        return None
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1]
    return _last_capital_letter(text)


def vote_del_tom(answers: Sequence[Optional[str]]) -> Optional[str]:
    """Protocol 3: majority vote, alphabetic tiebreak; ignores None."""
    valid = [a for a in answers if a in _VALID]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    winners = sorted(c for c, n in counts.items() if n == max_count)
    return winners[0]
```

- [ ] **Step 4: Run tests, expect pass**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_extractors.py -v
```

Expected: all 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/__init__.py scripts/eval/extractors.py \
        scripts/eval/tests/__init__.py scripts/eval/tests/test_extractors.py
git commit -m "feat(eval): answer extractors with 13 unit tests (TDD)"
```

---

### Task 2.2: TDD OpenAI-compatible clients

**Files:**
- Create: `scripts/eval/clients.py`
- Create: `scripts/eval/tests/test_clients.py`

- [ ] **Step 1: Write failing tests with mocked OpenAI client**

`scripts/eval/tests/test_clients.py`:

```python
from unittest.mock import MagicMock, patch
import pytest
from scripts.eval.clients import ChatClient, BackendSpec


def _mock_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)
    return resp


def test_chat_client_dashscope_uses_correct_base_url():
    spec = BackendSpec(name="dashscope", model="qwen3-8b")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("\\boxed{A}")
        client = ChatClient(spec, api_key="fake-key")
        client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
        MockOpenAI.assert_called_with(
            api_key="fake-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


def test_chat_client_deepseek_uses_correct_base_url():
    spec = BackendSpec(name="deepseek", model="deepseek-v4-pro")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("hi")
        client = ChatClient(spec, api_key="fake-key")
        client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        MockOpenAI.assert_called_with(api_key="fake-key", base_url="https://api.deepseek.com")


def test_chat_client_local_vllm_uses_provided_base_url():
    spec = BackendSpec(name="openai", model="qwen3-8b-tom",
                       base_url="http://localhost:8000/v1")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("ok")
        client = ChatClient(spec, api_key="dummy")
        client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        MockOpenAI.assert_called_with(api_key="dummy", base_url="http://localhost:8000/v1")


def test_chat_client_passes_extra_body_for_thinking():
    spec = BackendSpec(name="dashscope", model="qwen3-8b",
                       extra_body={"enable_thinking": False})
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("x")
        client = ChatClient(spec, api_key="fake")
        client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"enable_thinking": False}


def test_chat_client_retries_on_failure():
    spec = BackendSpec(name="deepseek", model="deepseek-v4-pro")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        mock_create = MockOpenAI.return_value.chat.completions.create
        mock_create.side_effect = [
            RuntimeError("transient"),
            RuntimeError("transient"),
            _mock_response("\\boxed{B}"),
        ]
        client = ChatClient(spec, api_key="fake", max_retries=3)
        result = client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        assert result.content == "\\boxed{B}"
        assert mock_create.call_count == 3


def test_chat_client_raises_after_max_retries():
    spec = BackendSpec(name="deepseek", model="deepseek-v4-pro")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("dead")
        client = ChatClient(spec, api_key="fake", max_retries=2)
        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "x"}], max_tokens=4)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_clients.py -v
```

Expected: ImportError (clients.py missing).

- [ ] **Step 3: Implement clients.py**

```python
"""OpenAI-compatible chat client for DashScope, DeepSeek, local vLLM, or generic OpenAI."""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


_BACKEND_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com",
}


@dataclass
class BackendSpec:
    name: str               # "dashscope" | "deepseek" | "openai"
    model: str              # model id
    base_url: Optional[str] = None  # required for "openai", else derived
    extra_body: dict = field(default_factory=dict)
    api_key_env: Optional[str] = None  # env var name for api key

    def resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        if self.name in _BACKEND_BASE_URLS:
            return _BACKEND_BASE_URLS[self.name]
        raise ValueError(f"backend {self.name!r} requires base_url")

    def resolve_api_key(self) -> str:
        if self.api_key_env:
            v = os.environ.get(self.api_key_env, "")
            if v:
                return v
        defaults = {
            "dashscope": "DASHSCOPE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env = defaults.get(self.name, "OPENAI_API_KEY")
        return os.environ.get(env, "")


@dataclass
class ChatResult:
    content: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient:
    """Thin wrapper around OpenAI client with retry + extra_body support."""

    def __init__(
        self,
        spec: BackendSpec,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        self.spec = spec
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = OpenAI(
            api_key=api_key if api_key is not None else spec.resolve_api_key(),
            base_url=spec.resolve_base_url(),
        )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 32,
        n: int = 1,
        extra_body_override: Optional[dict] = None,
    ) -> ChatResult:
        body = dict(self.spec.extra_body)
        if extra_body_override:
            body.update(extra_body_override)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=n,
                    timeout=self.timeout,
                    extra_body=body if body else None,
                )
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None) or type("U", (), {})()
                return ChatResult(
                    content=content,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt)
                    continue
        assert last_err is not None
        raise last_err
```

- [ ] **Step 4: Run tests, expect pass**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_clients.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/clients.py scripts/eval/tests/test_clients.py
git commit -m "feat(eval): OpenAI-compatible client with retry + 6 unit tests (TDD)"
```

---

### Task 2.3: TDD report aggregator

**Files:**
- Create: `scripts/eval/report.py`
- Create: `scripts/eval/tests/test_report.py`

- [ ] **Step 1: Write failing tests**

`scripts/eval/tests/test_report.py`:

```python
from scripts.eval.report import aggregate_results, format_markdown_table


def _sample_record(qid, lang, task, gold, pred, model, protocol):
    return {
        "question_id": qid,
        "language": lang,
        "task": task,
        "gold": gold,
        "pred": pred,
        "model": model,
        "protocol": protocol,
        "correct": pred == gold,
    }


def test_aggregate_basic_overall():
    rs = [
        _sample_record("1", "en", "False Belief", "A", "A", "m", "direct"),
        _sample_record("2", "en", "False Belief", "B", "C", "m", "direct"),
        _sample_record("3", "zh", "False Belief", "A", "A", "m", "direct"),
    ]
    agg = aggregate_results(rs)
    cell = agg[("m", "direct")]
    assert abs(cell["overall"] - 2/3) < 1e-6
    assert abs(cell["en"] - 1/2) < 1e-6
    assert abs(cell["zh"] - 1.0) < 1e-6


def test_aggregate_per_task_split():
    rs = [
        _sample_record("1", "en", "False Belief", "A", "A", "m", "direct"),
        _sample_record("2", "en", "Faux-pas", "B", "C", "m", "direct"),
        _sample_record("3", "en", "Faux-pas", "A", "A", "m", "direct"),
    ]
    agg = aggregate_results(rs)
    cell = agg[("m", "direct")]
    assert cell["task"]["False Belief"] == 1.0
    assert cell["task"]["Faux-pas"] == 0.5


def test_aggregate_multiple_models_protocols():
    rs = [
        _sample_record("1", "en", "False Belief", "A", "A", "m1", "direct"),
        _sample_record("1", "en", "False Belief", "A", "B", "m1", "cot"),
        _sample_record("1", "en", "False Belief", "A", "A", "m2", "direct"),
    ]
    agg = aggregate_results(rs)
    assert ("m1", "direct") in agg
    assert ("m1", "cot") in agg
    assert ("m2", "direct") in agg
    assert agg[("m1", "direct")]["overall"] == 1.0
    assert agg[("m1", "cot")]["overall"] == 0.0


def test_format_markdown_table_contains_headers():
    agg = {
        ("qwen3-8b-nt", "direct"): {
            "overall": 0.5349, "en": 0.55, "zh": 0.52,
            "task": {"False Belief": 0.6, "Faux-pas": 0.4},
        },
    }
    md = format_markdown_table(agg)
    assert "qwen3-8b-nt" in md
    assert "direct" in md
    assert "0.5349" in md
```

- [ ] **Step 2: Run tests, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_report.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement report.py**

```python
"""Aggregate per-question eval results into model-x-protocol-x-language-x-task table."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable


def aggregate_results(records: Iterable[dict]) -> dict:
    """Return dict {(model, protocol): {overall, en, zh, task: {<task>: acc}}}.

    Each record requires: model, protocol, language, task, correct (bool).
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["model"], r["protocol"])].append(r)

    result = {}
    for key, rs in by_key.items():
        n = len(rs)
        correct = sum(1 for r in rs if r["correct"])
        overall = correct / n if n else 0.0

        en_rs = [r for r in rs if r["language"] == "en"]
        zh_rs = [r for r in rs if r["language"] == "zh"]

        en_acc = sum(1 for r in en_rs if r["correct"]) / len(en_rs) if en_rs else 0.0
        zh_acc = sum(1 for r in zh_rs if r["correct"]) / len(zh_rs) if zh_rs else 0.0

        task_acc: dict[str, float] = {}
        task_groups: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            task_groups[r["task"]].append(r)
        for task, t_rs in task_groups.items():
            task_acc[task] = sum(1 for r in t_rs if r["correct"]) / len(t_rs)

        result[key] = {
            "overall": overall,
            "en": en_acc,
            "zh": zh_acc,
            "task": task_acc,
            "n": n,
        }
    return result


def format_markdown_table(agg: dict) -> str:
    """Format the main results table."""
    lines = ["| Model | Protocol | n | Overall | EN | ZH |",
             "|---|---|---|---|---|---|"]
    for (model, protocol), cell in sorted(agg.items()):
        lines.append(
            f"| {model} | {protocol} | {cell['n']} | "
            f"{cell['overall']:.4f} | {cell['en']:.4f} | {cell['zh']:.4f} |"
        )
    lines.append("")
    # Per-task breakdown
    lines.append("## Per-task breakdown")
    all_tasks: set[str] = set()
    for cell in agg.values():
        all_tasks.update(cell["task"].keys())
    sorted_tasks = sorted(all_tasks)
    header = "| Model | Protocol | " + " | ".join(sorted_tasks) + " |"
    sep = "|" + "---|" * (2 + len(sorted_tasks))
    lines.extend([header, sep])
    for (model, protocol), cell in sorted(agg.items()):
        row = f"| {model} | {protocol} |"
        for t in sorted_tasks:
            v = cell["task"].get(t)
            row += f" {v:.4f} |" if v is not None else " - |"
        lines.append(row)
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_report.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/report.py scripts/eval/tests/test_report.py
git commit -m "feat(eval): result aggregator + markdown table with 4 unit tests"
```

---

### Task 2.4: run_tombench.py main runner

**Files:**
- Create: `scripts/eval/run_tombench.py`
- Create: `scripts/eval/tests/test_run_tombench.py`

- [ ] **Step 1: Write failing test for prompt-building helpers**

`scripts/eval/tests/test_run_tombench.py`:

```python
from scripts.eval.run_tombench import (
    build_direct_messages,
    build_cot_messages,
    build_user_prompt_zh,
    build_user_prompt_en,
)


def test_build_user_prompt_en():
    text = build_user_prompt_en(
        story="Alice put marble in box.",
        question="Where does Bob look?",
        opt_a="box", opt_b="basket", opt_c="bag", opt_d="cup",
    )
    assert "Story:" in text
    assert "Alice put marble in box." in text
    assert "Where does Bob look?" in text
    assert "A. box" in text
    assert "D. cup" in text


def test_build_user_prompt_zh():
    text = build_user_prompt_zh(
        story="小明把球放进盒子。",
        question="小红会去哪里找？",
        opt_a="盒子", opt_b="篮子", opt_c="书包", opt_d="杯子",
    )
    assert "故事：" in text
    assert "小明把球放进盒子。" in text
    assert "A. 盒子" in text


def test_build_direct_messages_has_system_prompt():
    msgs = build_direct_messages(
        story="s", question="q",
        opt_a="a", opt_b="b", opt_c="c", opt_d="d",
        language="en",
    )
    assert msgs[0]["role"] == "system"
    assert "\\boxed{X}" in msgs[0]["content"]
    assert "Do not include any explanation" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_build_cot_messages_allows_thinking():
    msgs = build_cot_messages(
        story="s", question="q",
        opt_a="a", opt_b="b", opt_c="c", opt_d="d",
        language="en",
    )
    assert "Think step by step" in msgs[0]["content"]
    assert "\\boxed{X}" in msgs[0]["content"]
```

- [ ] **Step 2: Run, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_run_tombench.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement run_tombench.py — prompt helpers + main**

```python
"""Run ToMBench evaluation against any OpenAI-compatible chat backend."""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
from tqdm import tqdm

from scripts.eval.clients import BackendSpec, ChatClient, ChatResult
from scripts.eval.extractors import extract_direct, extract_cot, vote_del_tom
from scripts.eval.report import aggregate_results, format_markdown_table


SYSTEM_PROMPT_DIRECT = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Read the story and the question carefully, then output ONLY your final answer "
    "in the format \\boxed{X} where X is one of A, B, C, D. "
    "Do not include any explanation, reasoning, or extra text."
)

SYSTEM_PROMPT_COT = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Think step by step about the mental states of the characters, "
    "then output your final answer in the format \\boxed{X} where X is one of A, B, C, D. "
    "Put your final \\boxed{X} on the last line."
)


def build_user_prompt_en(*, story, question, opt_a, opt_b, opt_c, opt_d) -> str:
    return (
        f"Story:\n{story}\n\n"
        f"Question: {question}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
    )


def build_user_prompt_zh(*, story, question, opt_a, opt_b, opt_c, opt_d) -> str:
    return (
        f"故事：\n{story}\n\n"
        f"问题：{question}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
    )


def build_direct_messages(*, story, question, opt_a, opt_b, opt_c, opt_d, language: str) -> list[dict]:
    builder = build_user_prompt_zh if language == "zh" else build_user_prompt_en
    user = builder(story=story, question=question,
                   opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": user},
    ]


def build_cot_messages(*, story, question, opt_a, opt_b, opt_c, opt_d, language: str) -> list[dict]:
    builder = build_user_prompt_zh if language == "zh" else build_user_prompt_en
    user = builder(story=story, question=question,
                   opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user},
    ]


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------

def _cache_path(cache_dir: Path, model: str, protocol: str, qid: str, sample_idx: int = 0) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe_model}__{protocol}__{qid}__s{sample_idx}.json"


def _load_cached(p: Path) -> Optional[dict]:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _save_cached(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False))


# ----------------------------------------------------------------------
# Per-question evaluation
# ----------------------------------------------------------------------

def evaluate_one(
    *,
    client: ChatClient,
    record: dict,
    protocol: str,
    cache_dir: Path,
    model_id_for_cache: str,
) -> dict:
    qid = record["question_id"]
    language = record["language"]
    gold = record["gold"]
    task = record["task"]

    if protocol == "direct":
        messages = build_direct_messages(
            story=record["story"], question=record["question"],
            opt_a=record["opt_a"], opt_b=record["opt_b"],
            opt_c=record["opt_c"], opt_d=record["opt_d"],
            language=language,
        )
        sample_params = dict(temperature=0.0, top_p=1.0, max_tokens=32)
        n_samples = 1
    elif protocol == "cot":
        messages = build_cot_messages(
            story=record["story"], question=record["question"],
            opt_a=record["opt_a"], opt_b=record["opt_b"],
            opt_c=record["opt_c"], opt_d=record["opt_d"],
            language=language,
        )
        sample_params = dict(temperature=0.6, top_p=0.9, max_tokens=1024)
        n_samples = 1
    elif protocol == "del_tom":
        messages = build_cot_messages(
            story=record["story"], question=record["question"],
            opt_a=record["opt_a"], opt_b=record["opt_b"],
            opt_c=record["opt_c"], opt_d=record["opt_d"],
            language=language,
        )
        sample_params = dict(temperature=0.7, top_p=0.95, max_tokens=1024)
        n_samples = 8
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    answers: list[Optional[str]] = []
    raw_responses: list[str] = []
    for sample_idx in range(n_samples):
        cache_p = _cache_path(cache_dir, model_id_for_cache, protocol, qid, sample_idx)
        cached = _load_cached(cache_p)
        if cached is not None:
            content = cached["content"]
        else:
            res: ChatResult = client.chat(messages, **sample_params)
            content = res.content
            _save_cached(cache_p, {
                "qid": qid, "protocol": protocol, "sample_idx": sample_idx,
                "content": content,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            })
        raw_responses.append(content)
        if protocol == "direct":
            answers.append(extract_direct(content))
        else:
            answers.append(extract_cot(content))

    if protocol == "del_tom":
        pred = vote_del_tom(answers)
    else:
        pred = answers[0]

    return {
        "question_id": qid,
        "language": language,
        "task": task,
        "gold": gold,
        "pred": pred,
        "model": model_id_for_cache,
        "protocol": protocol,
        "correct": pred == gold,
        "raw_responses": raw_responses,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _build_backend(args) -> tuple[BackendSpec, str]:
    """Returns (spec, cache_id)."""
    if args.preset == "baseline-all":
        raise SystemExit("--preset baseline-all expands to multiple runs; use it via the runner script, not as a single backend.")
    if args.backend == "dashscope":
        extra: dict = {}
        if args.thinking is not None:
            extra["enable_thinking"] = args.thinking
        spec = BackendSpec(name="dashscope", model=args.model, extra_body=extra)
        cache_id = f"{args.model}-{'t' if args.thinking else 'nt'}"
    elif args.backend == "deepseek":
        spec = BackendSpec(name="deepseek", model=args.model)
        cache_id = args.model
    elif args.backend == "openai":
        if not args.base_url:
            raise SystemExit("--backend openai requires --base-url")
        spec = BackendSpec(name="openai", model=args.model, base_url=args.base_url)
        cache_id = args.model
    else:
        raise SystemExit(f"unknown backend: {args.backend}")
    return spec, cache_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["baseline-all"], default=None,
                   help="run a preset of multiple backends (baseline-all = qwen3-8b nt+t + deepseek-v4-pro)")
    p.add_argument("--backend", choices=["dashscope", "deepseek", "openai"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--thinking", type=lambda x: x.lower() == "true", default=None,
                   help="for dashscope qwen3-8b: enable_thinking true|false")
    p.add_argument("--base-url", default=None)
    p.add_argument("--protocols", default="direct",
                   help="comma-separated subset of direct,cot,del_tom")
    p.add_argument("--data", default="data/tom/tombench_eval.jsonl")
    p.add_argument("--output", default="output/eval/result.json")
    p.add_argument("--cache-dir", default="output/eval_cache")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only first N questions (debug)")
    args = p.parse_args()

    if args.preset == "baseline-all":
        return _run_baseline_all(args)

    # Single-backend run
    spec, cache_id = _build_backend(args)
    client = ChatClient(spec=spec)
    return _run_single(args, client, cache_id)


def _run_single(args, client: ChatClient, cache_id: str):
    data_path = Path(args.data)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with jsonlines.open(data_path) as reader:
        for r in reader:
            records.append(r)
    if args.limit:
        records = records[: args.limit]

    protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]
    all_results: list[dict] = []
    for protocol in protocols:
        print(f"=== {cache_id} :: {protocol} :: {len(records)} questions ===")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(evaluate_one,
                          client=client, record=r, protocol=protocol,
                          cache_dir=cache_dir, model_id_for_cache=cache_id)
                for r in records
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc=protocol):
                all_results.append(f.result())

    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"wrote {len(all_results)} eval records to {out_path}")

    agg = aggregate_results(all_results)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(format_markdown_table(agg))
    print(f"wrote markdown summary to {md_path}")


def _run_baseline_all(args):
    """Run Qwen3-8B (nt + t) and deepseek-v4-pro on both direct + cot."""
    plans = [
        dict(backend="dashscope", model="qwen3-8b", thinking=False, cache_id="qwen3-8b-nt"),
        dict(backend="dashscope", model="qwen3-8b", thinking=True,  cache_id="qwen3-8b-t"),
        dict(backend="deepseek",  model="deepseek-v4-pro", thinking=None, cache_id="deepseek-v4-pro"),
    ]

    args.protocols = "direct,cot"
    args.output = "output/eval/baseline_combined.json"

    combined: list[dict] = []
    for plan in plans:
        args.backend = plan["backend"]
        args.model = plan["model"]
        args.thinking = plan["thinking"]
        spec, _ = _build_backend(args)
        client = ChatClient(spec=spec)
        cache_id = plan["cache_id"]

        records = []
        with jsonlines.open(args.data) as reader:
            for r in reader:
                records.append(r)
        if args.limit:
            records = records[: args.limit]

        for protocol in args.protocols.split(","):
            protocol = protocol.strip()
            print(f"=== {cache_id} :: {protocol} :: {len(records)} questions ===")
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [
                    ex.submit(evaluate_one,
                              client=client, record=r, protocol=protocol,
                              cache_dir=Path(args.cache_dir),
                              model_id_for_cache=cache_id)
                    for r in records
                ]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"{cache_id}/{protocol}"):
                    combined.append(f.result())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    print(f"wrote {len(combined)} records to {out_path}")

    agg = aggregate_results(combined)
    md_path = Path("output/eval/baseline_report.md")
    md_path.write_text(format_markdown_table(agg))
    print(f"wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, expect pass**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/eval/tests/test_run_tombench.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/run_tombench.py scripts/eval/tests/test_run_tombench.py
git commit -m "feat(eval): main run_tombench runner with 3 protocols and caching"
```

---

## Phase 3 — ToMBench evaluation data

### Task 3.1: build_tombench_eval.py

**Files:**
- Create: `scripts/data/__init__.py` (empty)
- Create: `scripts/data/schema.py`
- Create: `scripts/data/build_tombench_eval.py`
- Create: `scripts/data/tests/__init__.py` (empty)
- Create: `scripts/data/tests/test_schema.py`

- [ ] **Step 1: Write schema dataclass**

`scripts/data/schema.py`:

```python
"""Unified data record schema (training + eval)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


# ToMBench's 20 ability strings → 8 broad task categories
ABILITY_TO_TASK = {
    "Belief: Location false beliefs": "False Belief",
    "Belief: Identity false beliefs": "False Belief",
    "Belief: Strange Story Task": "Strange Story",
    "Belief: Ambiguous Story": "Strange Story",
    "Belief: Unexpected Outcome": "Unexpected Outcome",
    "Belief: Persuasion Story": "Persuasion Story",
    "Belief: Knowledge-Attention Link": "Knowledge",
    "Belief: Knowledge-Pretend Play Link": "Knowledge",
    "Belief: Percepts-Knowledge Link": "Knowledge",
    "Desire: Multiple Desires": "Desire",
    "Desire: Discrepant Desires": "Desire",
    "Emotion: Moral Emotions": "Emotion",
    "Emotion: Discrepant Emotions": "Emotion",
    "Emotion: Hidden Emotions": "Emotion",
    "Emotion: Emotion Regulation": "Emotion",
    "Intention: Prediction of Actions": "Intention",
    "Intention: Discrepant Intentions": "Intention",
    "Intention: Completion of Failed Actions": "Intention",
    "Non-literal Comm: Hinting": "Non-literal Comm",
    "Non-literal Comm: Faux-pas Recognition": "Non-literal Comm",
    "Non-literal Comm: Scalar Implicature": "Non-literal Comm",
}


@dataclass
class TomRecord:
    """One record (eval or train) in unified schema."""
    question_id: str
    source: str                 # tombench | hi_tom | exploretom | simpletom | socialiqa | synth
    language: str               # en | zh
    task: str                   # one of the 8 ToMBench broad categories
    story: str
    question: str
    opt_a: str
    opt_b: str
    opt_c: str
    opt_d: str
    gold: str                   # "A" | "B" | "C" | "D"

    def to_jsonl_dict(self) -> dict:
        return asdict(self)


def ability_to_task(ability: str) -> str:
    """Map ToMBench 'ability' field to one of 8 broad tasks; default to 'Other'."""
    return ABILITY_TO_TASK.get(ability.strip(), "Other")
```

- [ ] **Step 2: Write tests for schema**

`scripts/data/tests/test_schema.py`:

```python
from scripts.data.schema import TomRecord, ability_to_task


def test_tom_record_to_jsonl_dict():
    r = TomRecord(
        question_id="q1", source="tombench", language="en", task="False Belief",
        story="s", question="q",
        opt_a="a", opt_b="b", opt_c="c", opt_d="d", gold="A",
    )
    d = r.to_jsonl_dict()
    assert d["question_id"] == "q1"
    assert d["gold"] == "A"


def test_ability_to_task_known():
    assert ability_to_task("Belief: Location false beliefs") == "False Belief"
    assert ability_to_task("Non-literal Comm: Hinting") == "Non-literal Comm"


def test_ability_to_task_unknown_returns_other():
    assert ability_to_task("Some unknown ability") == "Other"
```

- [ ] **Step 3: Run schema tests**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/data/tests/test_schema.py -v
```

Expected: 3 tests pass.

- [ ] **Step 4: Write build_tombench_eval.py**

```python
"""Download ToMBench from GitHub and convert to unified JSONL.

Outputs:
- data/tom/tombench_eval.jsonl  (one record per (question, language))
- data/tom/tombench_eval_subset500.jsonl  (random 500 for training-time eval)
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import jsonlines

from scripts.data.schema import TomRecord, ability_to_task


TOMBENCH_GITHUB_BASE = "https://raw.githubusercontent.com/zhchen18/ToMBench/main/data"
TOMBENCH_FILES = [
    "Ambiguous Story Task.jsonl",
    "Completion of Failed Actions.jsonl",
    "Discrepant Desires.jsonl",
    "Discrepant Emotions.jsonl",
    "Discrepant Intentions.jsonl",
    "Emotion Regulation.jsonl",
    "False Belief Task.jsonl",
    "Faux-pas Recognition Test.jsonl",
    "Hidden Emotions.jsonl",
    "Hinting Task Test.jsonl",
    "Knowledge-Attention Links.jsonl",
    "Knowledge-Pretend Play Links.jsonl",
    "Moral Emotions.jsonl",
    "Multiple Desires.jsonl",
    "Percepts-Knowledge Links.jsonl",
    "Persuasion Story Task.jsonl",
    "Prediction of Actions.jsonl",
    "Scalar Implicature Test.jsonl",
    "Strange Story Task.jsonl",
    "Unexpected Outcome Test.jsonl",
]


def download_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname in TOMBENCH_FILES:
        local = out_dir / fname
        if local.exists():
            paths.append(local)
            continue
        url = f"{TOMBENCH_GITHUB_BASE}/{fname.replace(' ', '%20')}"
        print(f"downloading {fname} ...")
        urlretrieve(url, local)
        paths.append(local)
    return paths


def transform_one(raw: dict, idx_in_file: int, fname: str) -> list[TomRecord]:
    """One raw ToMBench entry → 2 records (en + zh)."""
    ability = raw.get("能力\nABILITY", "")
    task = ability_to_task(ability)
    qid_base = f"{fname.replace('.jsonl','').replace(' ', '_')}_{idx_in_file}"
    gold = (raw.get("答案\nANSWER") or raw.get("ANSWER", "")).strip()
    if gold not in {"A", "B", "C", "D"}:
        return []

    records = []
    # English
    en_story = raw.get("STORY") or ""
    en_q = raw.get("QUESTION") or ""
    en_a = raw.get("OPTION-A") or ""
    en_b = raw.get("OPTION-B") or ""
    en_c = raw.get("OPTION-C") or ""
    en_d = raw.get("OPTION-D") or ""
    if en_story and en_q and en_a:
        records.append(TomRecord(
            question_id=f"{qid_base}_en", source="tombench",
            language="en", task=task,
            story=en_story, question=en_q,
            opt_a=en_a, opt_b=en_b, opt_c=en_c, opt_d=en_d,
            gold=gold,
        ))
    # Chinese
    zh_story = raw.get("故事") or ""
    zh_q = raw.get("问题") or ""
    zh_a = raw.get("选项A") or ""
    zh_b = raw.get("选项B") or ""
    zh_c = raw.get("选项C") or ""
    zh_d = raw.get("选项D") or ""
    if zh_story and zh_q and zh_a:
        records.append(TomRecord(
            question_id=f"{qid_base}_zh", source="tombench",
            language="zh", task=task,
            story=zh_story, question=zh_q,
            opt_a=zh_a, opt_b=zh_b, opt_c=zh_c, opt_d=zh_d,
            gold=gold,
        ))
    return records


def main():
    raw_dir = Path("data/tom/raw/tombench")
    out_full = Path("data/tom/tombench_eval.jsonl")
    out_sub = Path("data/tom/tombench_eval_subset500.jsonl")

    paths = download_all(raw_dir)
    all_records: list[TomRecord] = []
    for p in paths:
        with jsonlines.open(p) as reader:
            for idx, raw in enumerate(reader):
                all_records.extend(transform_one(raw, idx, p.name))

    out_full.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_full, "w") as w:
        for r in all_records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(all_records)} records to {out_full}")

    random.seed(42)
    subset = random.sample(all_records, k=min(500, len(all_records)))
    with jsonlines.open(out_sub, "w") as w:
        for r in subset:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(subset)} subset records to {out_sub}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the builder**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_tombench_eval.py
```

Expected: downloads 20 JSONL files, writes ~5000-5720 records (≈2860 questions × 2 languages, minus rows missing one language).

- [ ] **Step 6: Spot-check the output**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  bash -c "wc -l data/tom/tombench_eval.jsonl data/tom/tombench_eval_subset500.jsonl && head -1 data/tom/tombench_eval.jsonl | python -m json.tool"
```

Expected: full file has thousands of lines; subset has 500; first record has `question_id`, `gold` ∈ {A,B,C,D}, etc.

- [ ] **Step 7: Commit**

```bash
git add scripts/data/__init__.py scripts/data/schema.py \
        scripts/data/build_tombench_eval.py \
        scripts/data/tests/__init__.py scripts/data/tests/test_schema.py
git commit -m "feat(data): ToMBench eval downloader + unified schema + 3 unit tests"
```

---

### Task 3.2: Smoke-test eval framework on 10 questions

**Files:**
- (no new files)

- [ ] **Step 1: Run eval on 10 questions with Qwen3-8B (non-thinking)**

```bash
export DASHSCOPE_API_KEY=<your_key>
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/eval/run_tombench.py \
    --backend dashscope --model qwen3-8b --thinking false \
    --protocols direct \
    --limit 10 \
    --output output/eval/smoke.json \
    --concurrency 4
```

Expected: progress bar shows 10/10; writes `output/eval/smoke.json` and `output/eval/smoke.md`.

- [ ] **Step 2: Inspect output**

```bash
cat output/eval/smoke.md
```

Expected: a markdown table with one row showing accuracy on 10 questions (likely 0.4-0.8 range).

- [ ] **Step 3: Verify caching works**

Re-run the same command. Expected: progress bar completes instantly (all cache hits).

- [ ] **Step 4: No commit needed (smoke artifacts only)**

If smoke artifacts accidentally tracked:
```bash
git status
# if output/eval/smoke.* listed, do nothing — it's gitignored
```

---

## Phase 4 — Other data sources + dedupe

### Task 4.1: build_socialiqa.py

**Files:**
- Create: `scripts/data/build_socialiqa.py`

- [ ] **Step 1: Write the builder**

```python
"""Build ~1.5k records from SocialIQa (allenai/social_i_qa)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def main():
    random.seed(42)
    ds = load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
    # ds fields: context, question, answerA, answerB, answerC, label ('1'/'2'/'3')

    out = Path("data/tom/raw/socialiqa.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert 3-option to 4-option by adding "None of the above" as distractor
    # then shuffle to randomize gold position
    records: list[TomRecord] = []
    n = 1500
    for i, row in enumerate(ds.shuffle(seed=42).select(range(n * 2))):
        # Try to get 1500 well-formed; some may be skipped
        if len(records) >= n:
            break
        opts = [row["answerA"], row["answerB"], row["answerC"], "None of the above"]
        label = row["label"].strip()
        if label not in {"1", "2", "3"}:
            continue
        gold_idx = int(label) - 1
        # Shuffle 4 options
        idxs = list(range(4))
        random.shuffle(idxs)
        new_opts = [opts[j] for j in idxs]
        new_gold = "ABCD"[idxs.index(gold_idx)]
        rec = TomRecord(
            question_id=f"socialiqa_{i}",
            source="socialiqa", language="en", task="Other",
            story=row["context"], question=row["question"],
            opt_a=new_opts[0], opt_b=new_opts[1],
            opt_c=new_opts[2], opt_d=new_opts[3],
            gold=new_gold,
        )
        records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_socialiqa.py
```

Expected: ~1500 records to `data/tom/raw/socialiqa.jsonl`. First run downloads SocialIQa via HuggingFace datasets (~5 min).

- [ ] **Step 3: Commit**

```bash
git add scripts/data/build_socialiqa.py
git commit -m "feat(data): SocialIQa builder (1.5k records, 3→4 option transform)"
```

---

### Task 4.2: build_simpletom.py

**Files:**
- Create: `scripts/data/build_simpletom.py`

- [ ] **Step 1: Write builder**

```python
"""Build ~1k records from SimpleToM (allenai/simpletom)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def _transform_to_mcq(row: dict, idx: int) -> TomRecord | None:
    """SimpleToM has yes/no behavior-prediction; convert to 4-MCQ with synonyms."""
    story = row.get("story") or row.get("context") or ""
    question = row.get("question") or row.get("mental_state_question") or ""
    gold_bool = row.get("answer") or row.get("label")  # boolean or "yes"/"no"
    if isinstance(gold_bool, bool):
        gold_yes = gold_bool
    elif isinstance(gold_bool, str):
        gold_yes = gold_bool.lower().startswith("y")
    else:
        return None
    # 4 options for a yes/no flavor
    opts = ["Yes, they will", "No, they will not", "Cannot be determined", "Both yes and no"]
    gold_letter = "A" if gold_yes else "B"
    if not story or not question:
        return None
    return TomRecord(
        question_id=f"simpletom_{idx}",
        source="simpletom", language="en", task="Other",
        story=story, question=question,
        opt_a=opts[0], opt_b=opts[1], opt_c=opts[2], opt_d=opts[3],
        gold=gold_letter,
    )


def main():
    random.seed(42)
    out = Path("data/tom/raw/simpletom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("allenai/SimpleToM", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"SimpleToM not available via HF datasets: {e}")
        print("Skipping SimpleToM source; merge_and_dedupe will fall back to other sources.")
        with jsonlines.open(out, "w") as w:
            pass
        return

    records: list[TomRecord] = []
    for i, row in enumerate(ds.shuffle(seed=42).select(range(min(1500, len(ds))))):
        if len(records) >= 1000:
            break
        rec = _transform_to_mcq(row, i)
        if rec is not None:
            records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_simpletom.py
```

Expected: 1000 records or fewer; if HF dataset unavailable, writes empty file and continues (graceful).

- [ ] **Step 3: Commit**

```bash
git add scripts/data/build_simpletom.py
git commit -m "feat(data): SimpleToM builder with graceful HF fallback"
```

---

### Task 4.3: build_exploretom.py

**Files:**
- Create: `scripts/data/build_exploretom.py`

- [ ] **Step 1: Write builder**

```python
"""Build ~2k records from ExploreToM (facebookresearch/ExploreToM)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def main():
    random.seed(42)
    out = Path("data/tom/raw/exploretom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    candidate_repos = [
        ("facebook/ExploreToM", None),
        ("facebookresearch/ExploreToM", None),
    ]
    ds = None
    for repo, split in candidate_repos:
        try:
            ds = load_dataset(repo, split=split or "train", trust_remote_code=True)
            print(f"loaded {repo}")
            break
        except Exception as e:
            print(f"could not load {repo}: {e}")
    if ds is None:
        print("ExploreToM not available; writing empty file (will continue without this source)")
        with jsonlines.open(out, "w") as w:
            pass
        return

    n_target = 2000
    records: list[TomRecord] = []
    for i, row in enumerate(ds.shuffle(seed=42).select(range(min(len(ds), n_target * 2)))):
        if len(records) >= n_target:
            break
        story = row.get("story") or row.get("context", "")
        question = row.get("question", "")
        # Some ExploreToM splits provide options; if not, skip
        opts = row.get("options")
        gold = row.get("answer")
        if not (story and question and opts and gold):
            continue
        if isinstance(opts, list) and len(opts) >= 4:
            opt_list = list(opts[:4])
        else:
            continue
        # gold may be index, letter, or text
        gold_letter: str | None = None
        if isinstance(gold, int) and 0 <= gold < 4:
            gold_letter = "ABCD"[gold]
        elif isinstance(gold, str):
            if gold.strip().upper() in {"A", "B", "C", "D"}:
                gold_letter = gold.strip().upper()
            else:
                for j, o in enumerate(opt_list):
                    if str(o).strip() == gold.strip():
                        gold_letter = "ABCD"[j]
                        break
        if gold_letter is None:
            continue
        records.append(TomRecord(
            question_id=f"exploretom_{i}",
            source="exploretom", language="en", task="False Belief",
            story=story, question=question,
            opt_a=str(opt_list[0]), opt_b=str(opt_list[1]),
            opt_c=str(opt_list[2]), opt_d=str(opt_list[3]),
            gold=gold_letter,
        ))

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_exploretom.py
```

Expected: up to 2000 records; or graceful fallback.

- [ ] **Step 3: Commit**

```bash
git add scripts/data/build_exploretom.py
git commit -m "feat(data): ExploreToM builder with graceful HF fallback"
```

---

### Task 4.4: build_hitom.py

**Files:**
- Create: `scripts/data/build_hitom.py`

- [ ] **Step 1: Write builder (clones ToM-RL upstream)**

```python
"""Build ~2k records from Hi-ToM via the ToM-RL author's generation scripts.

Strategy:
1. Look for prebuilt Hi-ToM parquet in HF (YangXiao-nlp/DynToM or bigai-ai)
2. Else, fall back to a smaller hand-curated set from the original Hi-ToM repo
"""
from __future__ import annotations
import json
import random
import subprocess
from pathlib import Path

import jsonlines

from scripts.data.schema import TomRecord


def _try_hf() -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("YangXiao-nlp/Hi-ToM", split="train", trust_remote_code=True)
        return list(ds)
    except Exception as e:
        print(f"HF Hi-ToM not available: {e}")
        return []


def _clone_and_generate() -> list[dict]:
    """Clone the ToM-RL repo (which bundles Hi-ToM generators) and run them."""
    work = Path("data/tom/raw/hi_tom_gen")
    work.mkdir(parents=True, exist_ok=True)
    repo = work / "ToM-RL"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/bigai-ai/ToM-RL", str(repo)],
            check=True,
        )
    # The Hi-ToM generator is at repo / scripts / hitom (path varies between commits)
    # Look for generated parquet directly first:
    parquet = repo / "data" / "cleaned_tom" / "ToM_train_HiEx_hint.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(parquet)
        return [
            {col: table[col][i].as_py() for col in table.column_names}
            for i in range(table.num_rows)
        ]
    print("Hi-ToM parquet not found in cloned repo; returning empty.")
    return []


def _row_to_record(row: dict, idx: int) -> TomRecord | None:
    """Coerce a Hi-ToM raw row into TomRecord."""
    story = row.get("story") or row.get("context") or row.get("prompt", "")
    question = row.get("question") or row.get("query", "")
    options = row.get("options") or row.get("choices")
    gold = row.get("answer") or row.get("label")
    if not (story and question and options and gold is not None):
        return None
    if isinstance(options, list) and len(options) >= 4:
        opts = list(options[:4])
    elif isinstance(options, dict) and {"A", "B", "C", "D"}.issubset(options):
        opts = [options["A"], options["B"], options["C"], options["D"]]
    else:
        return None
    gold_letter: str | None = None
    if isinstance(gold, int):
        gold_letter = "ABCD"[gold] if 0 <= gold < 4 else None
    elif isinstance(gold, str):
        g = gold.strip().upper()
        if g in {"A", "B", "C", "D"}:
            gold_letter = g
        else:
            for j, o in enumerate(opts):
                if str(o).strip() == gold.strip():
                    gold_letter = "ABCD"[j]
                    break
    if gold_letter is None:
        return None
    return TomRecord(
        question_id=f"hitom_{idx}",
        source="hi_tom", language="en", task="False Belief",
        story=story, question=question,
        opt_a=str(opts[0]), opt_b=str(opts[1]),
        opt_c=str(opts[2]), opt_d=str(opts[3]),
        gold=gold_letter,
    )


def main():
    random.seed(42)
    out = Path("data/tom/raw/hi_tom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _try_hf() or _clone_and_generate()
    if not rows:
        print("Hi-ToM unavailable; writing empty file (other sources will compensate)")
        with jsonlines.open(out, "w") as w:
            pass
        return

    records: list[TomRecord] = []
    random.shuffle(rows)
    for i, row in enumerate(rows[:5000]):
        if len(records) >= 2000:
            break
        rec = _row_to_record(row, i)
        if rec is not None:
            records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_hitom.py
```

Expected: clones ToM-RL repo on first run; writes up to 2000 records.

- [ ] **Step 3: Commit**

```bash
git add scripts/data/build_hitom.py
git commit -m "feat(data): Hi-ToM builder via ToM-RL clone with HF fallback"
```

---

### Task 4.5: synth_tomtype.py (deepseek synthesis)

**Files:**
- Create: `scripts/data/synth_tomtype.py`
- Create: `scripts/data/tests/test_synth_parser.py`

- [ ] **Step 1: Write failing test for synth response parser**

`scripts/data/tests/test_synth_parser.py`:

```python
import json
from scripts.data.synth_tomtype import parse_synth_response


def test_parse_well_formed_json_object():
    raw = json.dumps({
        "story": "Alice sees a marble go into the basket.",
        "question": "Where will Bob look for the marble?",
        "options": {"A": "basket", "B": "box", "C": "cupboard", "D": "fridge"},
        "answer": "A",
    })
    rec = parse_synth_response(raw)
    assert rec is not None
    assert rec.story.startswith("Alice")
    assert rec.opt_a == "basket"
    assert rec.gold == "A"


def test_parse_extracts_json_from_markdown_fence():
    raw = "Here is the question:\n```json\n" + json.dumps({
        "story": "s", "question": "q",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer": "B",
    }) + "\n```"
    rec = parse_synth_response(raw)
    assert rec is not None
    assert rec.gold == "B"


def test_parse_returns_none_on_missing_field():
    raw = json.dumps({"story": "s", "question": "q", "options": {"A": "a"}})
    assert parse_synth_response(raw) is None


def test_parse_returns_none_on_garbage():
    assert parse_synth_response("not json at all") is None
```

- [ ] **Step 2: Run test, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/data/tests/test_synth_parser.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write synth_tomtype.py**

```python
"""Synthesize ToM MCQ questions via deepseek-v4-pro API.

Generates ~1.5k records covering the 8 ToMBench task types.
Each call asks for a fresh question + 4 options + gold letter.
Explicitly prohibits reproducing ToMBench questions.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
from openai import OpenAI
from tqdm import tqdm

from scripts.data.schema import TomRecord


SYNTH_SYSTEM = (
    "You are a careful question writer creating new theory-of-mind multiple-choice questions for training. "
    "Your output MUST be a single JSON object with keys: story, question, options (an object with A,B,C,D), answer (one of A,B,C,D). "
    "Do NOT reproduce, paraphrase, or translate any question from ToMBench by Chen et al. (ACL 2024). "
    "Write entirely new scenarios."
)


SYNTH_TASK_PROMPTS = {
    "False Belief":       "Write a False Belief task: a character's belief differs from reality after an unseen change.",
    "Strange Story":      "Write a Strange Story task involving subtle social misunderstanding or irony.",
    "Unexpected Outcome": "Write an Unexpected Outcome task where the result of an action differs from the character's expectation.",
    "Persuasion Story":   "Write a Persuasion Story task where one character tries to change another's belief.",
    "Knowledge":          "Write a Knowledge-Attention Link task where a character's knowledge depends on what they observed.",
    "Desire":             "Write a Multiple Desires task where two characters have different preferences.",
    "Emotion":            "Write a Discrepant Emotions task where two characters feel differently about the same event.",
    "Intention":          "Write a Prediction of Actions task asking what a character will do given their intention.",
    "Non-literal Comm":   "Write a Hinting Task: a character makes an indirect request and we must infer their actual desire.",
}

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_synth_response(text: str) -> Optional[TomRecord]:
    """Parse model output JSON into TomRecord, returning None on any failure."""
    if not text:
        return None
    # Strip markdown fences
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    # Find outermost JSON object
    m = _OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    try:
        story = obj["story"]
        question = obj["question"]
        opts = obj["options"]
        answer = obj["answer"]
    except (KeyError, TypeError):
        return None
    if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
        return None
    answer = str(answer).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return None
    return TomRecord(
        question_id="synth_pending",
        source="synth", language="en", task="Other",
        story=str(story), question=str(question),
        opt_a=str(opts["A"]), opt_b=str(opts["B"]),
        opt_c=str(opts["C"]), opt_d=str(opts["D"]),
        gold=answer,
    )


def call_deepseek_once(client: OpenAI, task: str) -> Optional[TomRecord]:
    user = SYNTH_TASK_PROMPTS[task] + " Output the JSON object directly."
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.9,
                max_tokens=800,
                timeout=60,
            )
            rec = parse_synth_response(resp.choices[0].message.content or "")
            if rec is None:
                continue
            rec.task = task
            return rec
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"synth call failed: {e}")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--task", default="all",
                   help="comma-separated subset of task types, or 'all'")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="data/tom/raw/synth.jsonl")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    if args.task == "all":
        tasks = list(SYNTH_TASK_PROMPTS.keys())
    else:
        tasks = [t.strip() for t in args.task.split(",")]
        for t in tasks:
            if t not in SYNTH_TASK_PROMPTS:
                raise SystemExit(f"unknown task: {t}")
    per_task = max(1, args.n // len(tasks))
    plan = []
    for t in tasks:
        plan.extend([t] * per_task)
    random.shuffle(plan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[TomRecord] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_deepseek_once, client, t) for t in plan]
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="synth")):
            rec = f.result()
            if rec is not None:
                rec.question_id = f"synth_{i}"
                records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} synthetic records to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/data/tests/test_synth_parser.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Run a small synthesis (50 records) to verify end-to-end**

```bash
export DEEPSEEK_API_KEY=<your_key>
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/synth_tomtype.py --n 50 --out data/tom/raw/synth_smoke.jsonl
```

Expected: ~30-50 records written (some calls may fail / parse-fail).

- [ ] **Step 6: Commit**

```bash
git add scripts/data/synth_tomtype.py scripts/data/tests/test_synth_parser.py
git commit -m "feat(data): deepseek synthesis with anti-leakage prompt + parser tests"
```

---

### Task 4.6: merge_and_dedupe.py (MinHash)

**Files:**
- Create: `scripts/data/merge_and_dedupe.py`
- Create: `scripts/data/tests/test_dedupe.py`

- [ ] **Step 1: Write failing test for MinHash similarity helper**

`scripts/data/tests/test_dedupe.py`:

```python
from scripts.data.merge_and_dedupe import jaccard_4gram, build_minhash_index


def test_jaccard_identical():
    assert jaccard_4gram("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert jaccard_4gram("hello world", "zzzzzzzz") < 0.1


def test_jaccard_partial_overlap():
    s = jaccard_4gram("the quick brown fox", "the quick brown dog")
    assert 0.2 < s < 0.9


def test_minhash_index_finds_near_duplicates():
    corpus = [
        ("a", "the cat sat on the mat"),
        ("b", "a different sentence entirely about dogs"),
        ("c", "the cat sat on the rug"),  # near-dup of a
    ]
    index = build_minhash_index(corpus)
    # Querying 'a' should return 'c' as candidate
    candidates = index.query("a", "the cat sat on the mat")
    assert "c" in candidates
    assert "b" not in candidates
```

- [ ] **Step 2: Run, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/data/tests/test_dedupe.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write merge_and_dedupe.py**

```python
"""Merge all data sources, dedupe internally (MinHash), and cross-check vs ToMBench eval.

Outputs:
- data/tom/tom_train.jsonl              (~8k training records)
- data/tom/tom_train_4k.jsonl           (random 4k subset for stage-1)
- data/tom/dedup_report.json            (audit: max-Jaccard distribution)
- docs/data-card.md                     (auto-generated)
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Iterable

import jsonlines
from datasketch import MinHash, MinHashLSH

from scripts.data.schema import TomRecord


# 4-gram tokenizer
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _4grams(text: str) -> set[str]:
    """Return set of word 4-grams for Jaccard / MinHash."""
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < 4:
        # Fall back to char 4-grams for very short strings
        chars = [c for c in text.lower() if not c.isspace()]
        return {"".join(chars[i:i+4]) for i in range(len(chars) - 3)} if len(chars) >= 4 else set(chars)
    return {" ".join(toks[i:i+4]) for i in range(len(toks) - 3)}


def jaccard_4gram(a: str, b: str) -> float:
    A, B = _4grams(a), _4grams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


class MinHashIndex:
    def __init__(self, threshold: float = 0.5, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.docs: dict[str, set[str]] = {}
        self.num_perm = num_perm

    def _mh(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        for g in _4grams(text):
            mh.update(g.encode("utf8"))
        return mh

    def add(self, key: str, text: str) -> None:
        self.docs[key] = _4grams(text)
        self.lsh.insert(key, self._mh(text))

    def query(self, exclude_key: str, text: str) -> list[str]:
        cands = self.lsh.query(self._mh(text))
        return [c for c in cands if c != exclude_key]


def build_minhash_index(corpus: Iterable[tuple[str, str]], threshold: float = 0.5) -> MinHashIndex:
    idx = MinHashIndex(threshold=threshold)
    for key, text in corpus:
        idx.add(key, text)
    return idx


def _text_for_match(rec: dict) -> str:
    """Canonical text used for similarity: question + 4 options."""
    return " ".join([
        rec["question"], rec["opt_a"], rec["opt_b"], rec["opt_c"], rec["opt_d"],
    ])


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with jsonlines.open(path) as r:
        return list(r)


def main():
    random.seed(42)
    raw_dir = Path("data/tom/raw")
    eval_path = Path("data/tom/tombench_eval.jsonl")
    out_full = Path("data/tom/tom_train.jsonl")
    out_4k = Path("data/tom/tom_train_4k.jsonl")
    report_path = Path("data/tom/dedup_report.json")

    sources = {
        "hi_tom":     raw_dir / "hi_tom.jsonl",
        "exploretom": raw_dir / "exploretom.jsonl",
        "simpletom":  raw_dir / "simpletom.jsonl",
        "socialiqa":  raw_dir / "socialiqa.jsonl",
        "synth":      raw_dir / "synth.jsonl",
    }

    # Step 1: Load ToMBench eval into MinHash index
    print("indexing ToMBench eval ...")
    eval_records = _load_raw(eval_path)
    eval_index = MinHashIndex(threshold=0.6)
    for r in eval_records:
        eval_index.add(r["question_id"], _text_for_match(r))

    # Step 2: For each source, drop any train record similar to any eval record
    max_jaccard_by_source: dict[str, list[float]] = {k: [] for k in sources}
    survivors: list[dict] = []
    dropped_by_leakage = 0
    for src, path in sources.items():
        rows = _load_raw(path)
        print(f"  {src}: loaded {len(rows)} rows from {path}")
        for r in rows:
            text = _text_for_match(r)
            cand = eval_index.query(exclude_key="", text=text)
            max_j = 0.0
            for c in cand:
                # Compute exact Jaccard for candidate
                j = jaccard_4gram(text, _text_for_match(next(e for e in eval_records if e["question_id"] == c)))
                if j > max_j:
                    max_j = j
            max_jaccard_by_source[src].append(max_j)
            if max_j > 0.6:
                dropped_by_leakage += 1
                continue
            survivors.append(r)

    print(f"  total after eval-leakage filter: {len(survivors)} (dropped {dropped_by_leakage})")

    # Step 3: Internal dedupe among survivors
    print("internal dedupe ...")
    internal_index = MinHashIndex(threshold=0.7)
    seen: list[dict] = []
    dropped_internal = 0
    for r in survivors:
        text = _text_for_match(r)
        dups = internal_index.query(exclude_key=r["question_id"], text=text)
        # Confirm with exact Jaccard
        is_dup = False
        for d in dups:
            other = next(s for s in seen if s["question_id"] == d)
            if jaccard_4gram(text, _text_for_match(other)) > 0.7:
                is_dup = True
                break
        if is_dup:
            dropped_internal += 1
            continue
        internal_index.add(r["question_id"], text)
        seen.append(r)

    print(f"  total after internal dedupe: {len(seen)} (dropped {dropped_internal})")

    # Step 4: Build messages field for ROLL training format
    from scripts.eval.run_tombench import SYSTEM_PROMPT_DIRECT, build_user_prompt_en, build_user_prompt_zh

    train_records: list[dict] = []
    for r in seen:
        builder = build_user_prompt_zh if r["language"] == "zh" else build_user_prompt_en
        user_text = builder(
            story=r["story"], question=r["question"],
            opt_a=r["opt_a"], opt_b=r["opt_b"],
            opt_c=r["opt_c"], opt_d=r["opt_d"],
        )
        train_records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
                {"role": "user", "content": user_text},
            ],
            "ground_truth": r["gold"],
            "tag": "tom_mcq",
            "source": r["source"],
            "language": r["language"],
            "task": r["task"],
            "question_id": r["question_id"],
        })

    out_full.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_full, "w") as w:
        for r in train_records:
            w.write(r)
    print(f"wrote {len(train_records)} train records to {out_full}")

    # Subset
    subset = random.sample(train_records, k=min(4000, len(train_records)))
    with jsonlines.open(out_4k, "w") as w:
        for r in subset:
            w.write(r)
    print(f"wrote {len(subset)} subset records to {out_4k}")

    # Dedup report
    report = {
        "n_total_survived": len(train_records),
        "n_dropped_by_eval_leakage": dropped_by_leakage,
        "n_dropped_by_internal_dedupe": dropped_internal,
        "per_source_max_jaccard_distribution": {
            src: {
                "mean": sum(v) / len(v) if v else 0.0,
                "max": max(v) if v else 0.0,
                "p95": sorted(v)[int(0.95 * len(v))] if v else 0.0,
                "n": len(v),
            }
            for src, v in max_jaccard_by_source.items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote dedup report to {report_path}")

    # Auto-generated data card
    card = Path("docs/data-card.md")
    lines = ["# Data Card", "", f"Generated from `merge_and_dedupe.py`.", ""]
    lines.append("## Sources after dedupe")
    by_src: dict[str, int] = {}
    for r in train_records:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    for src, n in sorted(by_src.items()):
        lines.append(f"- **{src}**: {n}")
    lines.append("")
    lines.append("## Leakage audit")
    lines.append(f"- Records dropped (>0.6 Jaccard with any ToMBench eval question): {dropped_by_leakage}")
    lines.append(f"- Records dropped (internal near-dup >0.7): {dropped_internal}")
    lines.append("")
    lines.append("## Per-source max-Jaccard vs ToMBench eval")
    lines.append("| Source | n | mean | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for src, st in report["per_source_max_jaccard_distribution"].items():
        lines.append(f"| {src} | {st['n']} | {st['mean']:.3f} | {st['p95']:.3f} | {st['max']:.3f} |")
    card.write_text("\n".join(lines))
    print(f"wrote data card to {card}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run dedupe tests**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest scripts/data/tests/test_dedupe.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Run merge end-to-end**

Prerequisite: all individual builders (Task 3.1, 4.1, 4.2, 4.3, 4.4, 4.5) have produced their raw files.

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/merge_and_dedupe.py
```

Expected:
- Prints per-source row counts
- Drops some records by leakage filter
- Writes `data/tom/tom_train.jsonl` (~6000-8000 records)
- Writes `data/tom/tom_train_4k.jsonl` (4000)
- Writes `data/tom/dedup_report.json`
- Writes `docs/data-card.md`

- [ ] **Step 6: Eyeball dedup report**

```bash
cat data/tom/dedup_report.json
cat docs/data-card.md
```

Expected: `per_source_max_jaccard_distribution[*].max ≤ 0.6` for every source.

- [ ] **Step 7: Manual sample check (50 synth records)**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev bash -c "\
  grep '\"source\": \"synth\"' data/tom/tom_train.jsonl | shuf -n 50 > /tmp/synth_sample.jsonl && cat /tmp/synth_sample.jsonl"
```

Eyeball: ensure no synth record reads like a verbatim ToMBench question.

- [ ] **Step 8: Commit**

```bash
git add scripts/data/merge_and_dedupe.py scripts/data/tests/test_dedupe.py docs/data-card.md
git commit -m "feat(data): MinHash dedupe + cross-set leakage filter + data card"
```

---

## Phase 5 — Baseline measurement

### Task 5.1: Run baseline-all

**Files:** none new; uses `run_tombench.py --preset baseline-all`.

- [ ] **Step 1: Run full baseline (3 models × 2 protocols × ~5720 questions)**

```bash
export DASHSCOPE_API_KEY=<key>
export DEEPSEEK_API_KEY=<key>
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/eval/run_tombench.py --preset baseline-all --concurrency 8
```

Expected: 6 progress bars (3 models × 2 protocols), each iterating ~5720 questions. Time: ~30-60 min depending on concurrency and API latency.

- [ ] **Step 2: Inspect baseline report**

```bash
cat output/eval/baseline_report.md
```

Expected: 6 rows (3 models × 2 protocols) plus per-task breakdown table. Record the values:
- `Y_base_nt = <number>` (qwen3-8b-nt direct overall)
- `Y_base_t = <number>` (qwen3-8b-t direct overall)
- `X = <number>` (deepseek-v4-pro direct overall)

- [ ] **Step 3: Go/No-go check**

Verify:
- All 6 rows have `n > 5000` (no widespread API failures)
- `Y_base_nt` ∈ [0.50, 0.80] (sanity bound)
- If `X < Y_base_nt`: STOP, debug protocol/prompt/extraction before proceeding

- [ ] **Step 4: Commit baseline report**

```bash
git add output/eval/baseline_report.md
# Note: this is normally gitignored. Force-add for archival.
git add -f output/eval/baseline_report.md
git commit -m "data(eval): baseline ToMBench scores (qwen3-8b nt+t, deepseek-v4-pro)"
```

---

## Phase 6 — ROLL reward worker (TDD)

### Task 6.1: TDD tom_mcq_reward_worker.py

**Files:**
- Create: `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py`
- Create: `framework/ROLL/tests/test_tom_mcq_reward.py`

- [ ] **Step 1: Read existing worker for reference**

```bash
cat framework/ROLL/roll/pipeline/rlvr/rewards/multiple_choice_boxed_rule_reward_worker.py | head -50
```

- [ ] **Step 2: Write the failing unit tests**

`framework/ROLL/tests/test_tom_mcq_reward.py`:

```python
"""Unit tests for TomMcqRewardWorker reward computation.

We test the pure-function reward components directly (not the full worker
class, which requires Ray cluster). The worker's compute_rewards method
simply applies these functions sample-wise.
"""
import math
import pytest

from roll.pipeline.rlvr.rewards.tom_mcq_reward_worker import (
    extract_boxed_letter,
    sigmoid_window,
    tom_mcq_reward_fn,
)


# ----- extract_boxed_letter -----

def test_extract_boxed_basic():
    letter, fmt_ok = extract_boxed_letter("\\boxed{A}")
    assert letter == "A"
    assert fmt_ok is True


def test_extract_boxed_in_text():
    letter, fmt_ok = extract_boxed_letter("My answer is \\boxed{C}")
    assert letter == "C"
    assert fmt_ok is True


def test_extract_no_boxed():
    letter, fmt_ok = extract_boxed_letter("just text no boxed")
    assert fmt_ok is False


def test_extract_invalid_letter_inside_box():
    # \boxed{Z} is invalid → fmt_ok False
    letter, fmt_ok = extract_boxed_letter("\\boxed{Z}")
    assert fmt_ok is False


# ----- sigmoid_window -----

def test_sigmoid_window_center_high():
    # In the middle of the window → near 1
    v = sigmoid_window(100, l_min=8, l_max=256, k=50)
    assert v > 0.9


def test_sigmoid_window_below_min_low():
    v = sigmoid_window(2, l_min=8, l_max=256, k=50)
    assert v < 0.2


def test_sigmoid_window_above_max_low():
    v = sigmoid_window(1000, l_min=8, l_max=256, k=50)
    assert v < 0.1


# ----- tom_mcq_reward_fn (composite) -----

def test_reward_correct_short_format():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=5,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    assert r_len > 0.0  # short but in window
    assert r_total > 0.0


def test_reward_correct_but_overlong_low_reward():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=2000,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    assert r_len < 0.1
    assert r_total < 0.1


def test_reward_wrong_answer_zero():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{B}", response_token_count=5,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 0.0
    assert r_total == 0.0


def test_reward_no_boxed_zero():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="The answer is A.", response_token_count=5,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 0.0
    assert r_total == 0.0


def test_reward_boundaries():
    # L = l_min boundary
    _, _, r_len_a, _ = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=8,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    # L = l_max boundary
    _, _, r_len_b, _ = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=256,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert 0.4 < r_len_a < 0.6  # roughly sigmoid(0) on the rise side
    assert 0.4 < r_len_b < 0.6  # roughly sigmoid(0) on the fall side


def test_reward_chinese_response():
    # Chinese response still works as long as \boxed{X} present
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="答案是 \\boxed{C}", response_token_count=10,
        ground_truth="C", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    assert r_total > 0.0
```

- [ ] **Step 3: Run, expect failure**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest framework/ROLL/tests/test_tom_mcq_reward.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement tom_mcq_reward_worker.py**

```python
"""ToM MCQ reward worker for ROLL RLVR pipeline.

Implements the L2 reward:
    R = R_fmt × R_out × R_len

- R_fmt: 1 if response contains \boxed{X} with X ∈ {A,B,C,D}, else 0
- R_out: 1 if extracted letter == ground_truth, else 0
- R_len: sigmoid window over response token count (encourages short, on-format answers)

The pure functions (extract_boxed_letter, sigmoid_window, tom_mcq_reward_fn) are
exposed at module level so they can be unit-tested without Ray.
"""
from __future__ import annotations
import json
import math
import re
from typing import Tuple

import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.logging import get_logger

logger = get_logger()

_BOXED = re.compile(r"\\boxed\{([A-D])\}")
_VALID_LETTERS = {"A", "B", "C", "D"}


def extract_boxed_letter(response: str) -> Tuple[str, bool]:
    """Return (letter, format_ok). Format_ok requires a valid \\boxed{[A-D]}."""
    if not response:
        return "", False
    m = _BOXED.search(response)
    if m:
        return m.group(1), True
    return "", False


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid_window(L: float, l_min: float, l_max: float, k: float) -> float:
    """Smooth window: rises near l_min, falls near l_max, plateau in between.

    R_len(L) = sigmoid(k * (L - l_min) / (l_max - l_min))
             * (1 - sigmoid(k * (L - l_max) / (l_max - l_min)))
    """
    span = max(1.0, l_max - l_min)
    rise = _sigmoid(k * (L - l_min) / span)
    fall = 1.0 - _sigmoid(k * (L - l_max) / span)
    return rise * fall


def tom_mcq_reward_fn(
    response: str,
    response_token_count: int,
    ground_truth: str,
    l_min: float = 8.0,
    l_max: float = 256.0,
    k: float = 50.0,
) -> Tuple[float, float, float, float]:
    """Compute (r_fmt, r_out, r_len, r_total) for a single response."""
    letter, fmt_ok = extract_boxed_letter(response)
    r_fmt = 1.0 if fmt_ok else 0.0
    r_out = 1.0 if (fmt_ok and letter == ground_truth) else 0.0
    r_len = sigmoid_window(float(response_token_count), l_min, l_max, k)
    r_total = r_fmt * r_out * r_len
    return r_fmt, r_out, r_len, r_total


class TomMcqRewardWorker(Worker):
    """RLVR reward worker for ToM MCQ with R_fmt × R_out × R_len reward."""

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(
            model_args=self.worker_config.model_args
        )
        # Hyperparams from worker_config (with defaults)
        self.l_min = float(getattr(self.worker_config, "l_min", 8))
        self.l_max = float(getattr(self.worker_config, "l_max", 256))
        self.k = float(getattr(self.worker_config, "k", 50))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        response_text_list = self.tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        ground_truths = data.non_tensor_batch["ground_truth"]

        scores: list[float] = []
        r_fmt_list: list[float] = []
        r_out_list: list[float] = []
        r_len_list: list[float] = []
        response_lengths: list[int] = []

        for i, (resp_tokens, gold) in enumerate(zip(data.batch["responses"], ground_truths)):
            response_text = response_text_list[i]
            # Effective response length: count non-pad tokens
            non_pad = (resp_tokens != self.tokenizer.pad_token_id).sum().item() \
                if self.tokenizer.pad_token_id is not None else len(resp_tokens)
            r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
                response=response_text,
                response_token_count=non_pad,
                ground_truth=str(gold),
                l_min=self.l_min, l_max=self.l_max, k=self.k,
            )
            scores.append(r_total)
            r_fmt_list.append(r_fmt)
            r_out_list.append(r_out)
            r_len_list.append(r_len)
            response_lengths.append(non_pad)

            try:
                letter, _ = extract_boxed_letter(response_text)
                self.logger.debug(json.dumps({
                    "r_fmt": r_fmt, "r_out": r_out, "r_len": r_len, "r_total": r_total,
                    "response_length": non_pad,
                    "extracted_letter": letter,
                    "ground_truth": str(gold),
                }, ensure_ascii=False))
            except Exception as e:
                self.logger.error(f"logging error: {e}")

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)

        # Additional aggregate metrics for tensorboard
        n = max(1, len(scores))
        metrics = {
            "reward/r_fmt_mean": sum(r_fmt_list) / n,
            "reward/r_out_mean": sum(r_out_list) / n,
            "reward/r_len_mean": sum(r_len_list) / n,
            "reward/r_total_mean": sum(scores) / n,
            "reward/response_length_mean": sum(response_lengths) / n,
        }

        output = DataProto.from_dict(tensors={
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": scores_tensor,
            "scores": scores_tensor,
        })
        output.meta_info = {"metrics": metrics}
        return output
```

- [ ] **Step 5: Run tests, expect pass**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  pytest framework/ROLL/tests/test_tom_mcq_reward.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 6: Commit**

```bash
git add framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py \
        framework/ROLL/tests/test_tom_mcq_reward.py
git commit -m "feat(roll-reward): TomMcqRewardWorker with R_fmt × R_out × R_len + 11 tests"
```

---

## Phase 7 — ROLL configuration files

### Task 7.1: rlvr_config_stage1.yaml

**Files:**
- Create: `configs/tombench-rlvr/rlvr_config_stage1.yaml`

- [ ] **Step 1: Write stage-1 config (small-scale, 4k × 200 steps)**

```yaml
# Stage-1: small-scale verification (4k data × 200 steps)
hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "qwen3-8B-tombench-rlvr-stage1"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
system_envs:
  USE_MODELSCOPE: '1'

checkpoint_config:
  type: file_system
  output_dir: /mnt/output/${exp_name}

track_with: tensorboard
tracker_kwargs:
  log_dir: /mnt/output/tensorboard/${exp_name}

num_gpus_per_node: 8

max_steps: 200
save_steps: 50
logging_steps: 1
eval_steps: 50
resume_from_checkpoint: false

rollout_batch_size: 64
prompt_length: 2048
response_length: 256

num_return_sequences_in_group: 8
ppo_epochs: 1
adv_estimator: "grpo"

# DAPO Clip-Higher
use_pg_clip_range: true
pg_clip_low: 0.20
pg_clip_high: 0.28
dual_clip_loss: true

value_clip: 0.5
reward_clip: 5
advantage_clip: 2.0

norm_mean_type: ~
norm_std_type: ~

max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95
error_max_len_clip: false

difficulty_loss_weight: false
length_loss_weight: false

add_token_level_kl: false

whiten_advantages: true

# DAPO Dynamic Sampling
use_additional_prompts: true
max_running_requests: 256
is_num_return_sequences_expand: false

pretrain: Qwen/Qwen3-8B
reward_pretrain: Qwen/Qwen3-8B

validation:
  data_args:
    template: qwen3
    file_name:
      - /mnt/data/tombench_eval_subset500.jsonl
  generating_args:
    max_new_tokens: 64
    top_p: 1.0
    top_k: -1
    num_beams: 1
    temperature: 0.0
    num_return_sequences: 1

actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50
  data_args:
    template: qwen3
    file_name:
      - /mnt/data/tom_train_4k.jsonl
    domain_interleave_probs:
      tom_mcq: 1.0
    dataset_dir: /mnt/data
    messages: messages
    interleave_probs: "1.0"
    preprocessing_num_workers: 16
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,16))
  infer_batch_size: 4

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.95
    top_k: 50
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 4096
  device_mapping: list(range(0,12))
  infer_batch_size: 1

reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: megatron_infer
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
  device_mapping: list(range(0,16))
  infer_batch_size: 4

rewards:
  tom_mcq:
    worker_cls: roll.pipeline.rlvr.rewards.tom_mcq_reward_worker.TomMcqRewardWorker
    tag_included: [tom_mcq]
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen3
    world_size: 8
    infer_batch_size: 16
    l_min: 8
    l_max: 256
    k: 50
```

- [ ] **Step 2: Validate YAML syntax**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python -c "import yaml; yaml.safe_load(open('configs/tombench-rlvr/rlvr_config_stage1.yaml'))"
```

Expected: no output (valid).

- [ ] **Step 3: Commit**

```bash
git add configs/tombench-rlvr/rlvr_config_stage1.yaml
git commit -m "config(roll): stage-1 RLVR config (4k × 200 steps, GRPO+Clip-Higher)"
```

---

### Task 7.2: rlvr_config_stage2.yaml

**Files:**
- Create: `configs/tombench-rlvr/rlvr_config_stage2.yaml`

- [ ] **Step 1: Write stage-2 config**

Use the same content as stage-1 with three differences:
- `exp_name: "qwen3-8B-tombench-rlvr-stage2"`
- `max_steps: 500`
- `save_steps: 100`
- Training data file: `/mnt/data/tom_train.jsonl` (full 8k) instead of `tom_train_4k.jsonl`

Full file (write it explicitly so engineers don't need to diff):

```yaml
# Stage-2: main training (8k data × 500 steps)
hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "qwen3-8B-tombench-rlvr-stage2"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
system_envs:
  USE_MODELSCOPE: '1'

checkpoint_config:
  type: file_system
  output_dir: /mnt/output/${exp_name}

track_with: tensorboard
tracker_kwargs:
  log_dir: /mnt/output/tensorboard/${exp_name}

num_gpus_per_node: 8

max_steps: 500
save_steps: 100
logging_steps: 1
eval_steps: 50
resume_from_checkpoint: false

rollout_batch_size: 64
prompt_length: 2048
response_length: 256

num_return_sequences_in_group: 8
ppo_epochs: 1
adv_estimator: "grpo"

use_pg_clip_range: true
pg_clip_low: 0.20
pg_clip_high: 0.28
dual_clip_loss: true

value_clip: 0.5
reward_clip: 5
advantage_clip: 2.0

max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95
error_max_len_clip: false

add_token_level_kl: false
whiten_advantages: true

use_additional_prompts: true
max_running_requests: 256
is_num_return_sequences_expand: false

pretrain: Qwen/Qwen3-8B
reward_pretrain: Qwen/Qwen3-8B

validation:
  data_args:
    template: qwen3
    file_name:
      - /mnt/data/tombench_eval_subset500.jsonl
  generating_args:
    max_new_tokens: 64
    top_p: 1.0
    top_k: -1
    num_beams: 1
    temperature: 0.0
    num_return_sequences: 1

actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50
  data_args:
    template: qwen3
    file_name:
      - /mnt/data/tom_train.jsonl
    domain_interleave_probs:
      tom_mcq: 1.0
    dataset_dir: /mnt/data
    messages: messages
    interleave_probs: "1.0"
    preprocessing_num_workers: 16
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,16))
  infer_batch_size: 4

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.95
    top_k: 50
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 4096
  device_mapping: list(range(0,12))
  infer_batch_size: 1

reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: megatron_infer
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
  device_mapping: list(range(0,16))
  infer_batch_size: 4

rewards:
  tom_mcq:
    worker_cls: roll.pipeline.rlvr.rewards.tom_mcq_reward_worker.TomMcqRewardWorker
    tag_included: [tom_mcq]
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen3
    world_size: 8
    infer_batch_size: 16
    l_min: 8
    l_max: 256
    k: 50
```

- [ ] **Step 2: Validate YAML**

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python -c "import yaml; yaml.safe_load(open('configs/tombench-rlvr/rlvr_config_stage2.yaml'))"
```

- [ ] **Step 3: Commit**

```bash
git add configs/tombench-rlvr/rlvr_config_stage2.yaml
git commit -m "config(roll): stage-2 RLVR config (8k × 500 steps)"
```

---

### Task 7.3: rlvr_config_stage3_l3.yaml (placeholder)

**Files:**
- Create: `configs/tombench-rlvr/rlvr_config_stage3_l3.yaml`

- [ ] **Step 1: Write L3 fallback config skeleton**

```yaml
# Stage-3 L3 fallback: same as stage-2 but with process-reward (R_struct + R_content).
# REQUIRES: R_content reward model trained first (see docs/runbook.md L3 section).
# Skeleton; activate only if Y'_direct < X − 0.02 after stage-2.

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "qwen3-8B-tombench-rlvr-stage3-l3"
seed: 42
logging_dir: ./output/logs
output_dir: ./output

checkpoint_config:
  type: file_system
  output_dir: /mnt/output/${exp_name}

# Identical to stage-2 except:
# 1. Reward worker is replaced with TomMcqL3RewardWorker (NOT YET IMPLEMENTED)
# 2. Two extra workers: deepseek_judge (R_struct) and rm_content (R_content via Qwen3-4B+LoRA)
#
# Implementation: see scripts/deploy/build_l3_components.md (to be added when L3 is triggered).
# For now this file is a stub so the Makefile target exists.

# When activating, copy stage-2 config above and add the L3 reward stack here.
```

- [ ] **Step 2: Commit**

```bash
git add configs/tombench-rlvr/rlvr_config_stage3_l3.yaml
git commit -m "config(roll): stage-3 L3 fallback skeleton (activated only if stage-2 misses target)"
```

---

## Phase 8 — TRAIN Docker images

### Task 8.1: docker/train/Dockerfile

**Files:**
- Create: `docker/train/Dockerfile`
- Create: `docker/train/entrypoint.sh`
- Create: `docker/train/docker-compose.yml`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# TRAIN image — runs on remote GPU server (16×H800)
# Based on NVIDIA's PyTorch container which already has CUDA + PyTorch matched.
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install ROLL framework requirements
COPY framework/ROLL/requirements_torch280_vllm.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

# Install ROLL in editable mode (source is mounted at runtime)
# We DON'T `pip install -e .` here because the source is mounted; instead
# we set PYTHONPATH at runtime via entrypoint.

ENV PYTHONPATH=/workspace:/workspace/framework/ROLL

# vLLM, megatron, flash-attn are typically already in the base image or in req.txt
# Verify imports during build:
RUN python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)" || \
    echo "vllm not yet importable; will resolve at runtime"

COPY docker/train/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
```

- [ ] **Step 2: Write entrypoint.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

# entrypoint for TRAIN docker
# Required env vars:
#   STAGE = stage1 | stage2 | stage3_l3
#
# Mounts (provided by docker-compose):
#   /workspace             — repo (read/write)
#   /mnt/data              — training data
#   /mnt/models            — model cache
#   /mnt/output            — training outputs

STAGE="${STAGE:-stage1}"
CONFIG_DIR="/workspace/configs/tombench-rlvr"
CONFIG_NAME="rlvr_config_${STAGE}"

echo "=========================================="
echo "TRAIN entrypoint"
echo "  stage:  ${STAGE}"
echo "  config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "  CUDA visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo "  GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================="

# Verify data + models present
test -f "/mnt/data/tom_train.jsonl" || { echo "ERROR: /mnt/data/tom_train.jsonl missing"; exit 1; }
test -f "/mnt/data/tombench_eval_subset500.jsonl" || { echo "ERROR: subset500 missing"; exit 1; }

# Install ROLL in editable mode (idempotent)
pip install -e /workspace/framework/ROLL >/dev/null

# Run training
cd /workspace/framework/ROLL
exec python examples/start_rlvr_pipeline.py \
  --config_path "${CONFIG_DIR}" \
  --config_name "${CONFIG_NAME}"
```

- [ ] **Step 3: Write docker-compose.yml**

```yaml
services:
  train:
    build:
      context: ../..
      dockerfile: docker/train/Dockerfile
    image: qwen3-tom-train:latest
    working_dir: /workspace
    volumes:
      - ../..:/workspace
      - ${TRAIN_DATA_DIR:-/data/cpfs_0/tom-data}:/mnt/data
      - ${TRAIN_MODELS_DIR:-/data/cpfs_0/models}:/mnt/models
      - ${TRAIN_OUTPUT_DIR:-/data/cpfs_0/tom-output}:/mnt/output
    environment:
      STAGE: ${STAGE:-stage1}
      USE_MODELSCOPE: '1'
      PYTHONPATH: /workspace:/workspace/framework/ROLL
      HF_HOME: /mnt/models/.cache/huggingface
      MODELSCOPE_CACHE: /mnt/models/.cache/modelscope
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    shm_size: 64gb
```

- [ ] **Step 4: Commit (build can't happen on DEV macOS; deferred to TRAIN)**

```bash
git add docker/train/Dockerfile docker/train/entrypoint.sh docker/train/docker-compose.yml
git commit -m "build(docker): TRAIN image (nvcr pytorch 24.10 + ROLL editable install)"
```

---

### Task 8.2: docker/serve/

**Files:**
- Create: `docker/serve/Dockerfile`
- Create: `docker/serve/entrypoint.sh`
- Create: `docker/serve/docker-compose.yml`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# SERVE image — vLLM OpenAI-compatible server for the trained Qwen3-8B-tom model.
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# vLLM with OpenAI API server
RUN pip install "vllm>=0.6.0"

COPY docker/serve/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/workspace/entrypoint.sh"]
```

- [ ] **Step 2: Write entrypoint.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/output/final_model}"
PORT="${SERVE_PORT:-8000}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen3-8b-tom}"

echo "=========================================="
echo "SERVE entrypoint"
echo "  model: ${MODEL_PATH}"
echo "  port:  ${PORT}"
echo "  name:  ${SERVED_NAME}"
echo "=========================================="

test -d "${MODEL_PATH}" || { echo "ERROR: model dir ${MODEL_PATH} missing"; exit 1; }

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --served-model-name "${SERVED_NAME}"
```

- [ ] **Step 3: Write docker-compose.yml**

```yaml
services:
  serve:
    build:
      context: ../..
      dockerfile: docker/serve/Dockerfile
    image: qwen3-tom-serve:latest
    working_dir: /workspace
    volumes:
      - ${TRAIN_OUTPUT_DIR:-/data/cpfs_0/tom-output}:/mnt/output
    environment:
      MODEL_PATH: /mnt/output/final_model
      SERVE_PORT: ${SERVE_PORT:-8000}
      SERVED_MODEL_NAME: qwen3-8b-tom
    ports:
      - "${SERVE_PORT:-8000}:${SERVE_PORT:-8000}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ipc: host
    shm_size: 16gb
```

- [ ] **Step 4: Commit**

```bash
git add docker/serve/Dockerfile docker/serve/entrypoint.sh docker/serve/docker-compose.yml
git commit -m "build(docker): SERVE image (vLLM OpenAI-compat for trained model)"
```

---

## Phase 9 — Deploy scripts

### Task 9.1: sync_to_train.sh / sync_from_train.sh

**Files:**
- Create: `scripts/deploy/sync_to_train.sh`
- Create: `scripts/deploy/sync_from_train.sh`

- [ ] **Step 1: Write sync_to_train.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Sync DEV → TRAIN: code + data
# Requires: configs/deploy.env with TRAIN_HOST, TRAIN_PATH, TRAIN_SSH_KEY, TRAIN_DATA_DIR

if [ ! -f configs/deploy.env ]; then
  echo "ERROR: configs/deploy.env missing. Copy from configs/deploy.env.example and fill in."
  exit 1
fi
source configs/deploy.env

# 1. Code + configs (excludes output/data/.git/cache)
echo "[sync-up] syncing code → ${TRAIN_HOST}:${TRAIN_PATH}/"
rsync -avz --delete \
  --exclude=output \
  --exclude=data \
  --exclude=.git \
  --exclude='**/__pycache__' \
  --exclude='*.pyc' \
  --exclude='**/.DS_Store' \
  --exclude='**/node_modules' \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  ./ "${TRAIN_HOST}:${TRAIN_PATH}/"

# 2. Training data (no --delete; manual cleanup if needed)
echo "[sync-up] syncing data → ${TRAIN_HOST}:${TRAIN_DATA_DIR}/"
ssh -i "${TRAIN_SSH_KEY}" "${TRAIN_HOST}" "mkdir -p ${TRAIN_DATA_DIR}"
rsync -avz --progress \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  ./data/tom/ "${TRAIN_HOST}:${TRAIN_DATA_DIR}/"

echo "[sync-up] done"
```

- [ ] **Step 2: Write sync_from_train.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Sync TRAIN → DEV: best checkpoint + tensorboard logs + eval results
if [ ! -f configs/deploy.env ]; then
  echo "ERROR: configs/deploy.env missing"
  exit 1
fi
source configs/deploy.env

mkdir -p output/checkpoints output/tensorboard output/eval

echo "[sync-down] best checkpoint ..."
rsync -avz --progress \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  "${TRAIN_HOST}:${TRAIN_OUTPUT_DIR}/best_checkpoint/" \
  ./output/checkpoints/best/ || echo "(no best_checkpoint yet)"

echo "[sync-down] tensorboard logs ..."
rsync -avz \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  "${TRAIN_HOST}:${TRAIN_OUTPUT_DIR}/tensorboard/" \
  ./output/tensorboard/ || echo "(no tb logs)"

echo "[sync-down] eval results ..."
rsync -avz \
  -e "ssh -i ${TRAIN_SSH_KEY}" \
  "${TRAIN_HOST}:${TRAIN_OUTPUT_DIR}/eval/" \
  ./output/eval/ || echo "(no eval results)"

echo "[sync-down] done"
```

- [ ] **Step 3: Make executable + commit**

```bash
chmod +x scripts/deploy/sync_to_train.sh scripts/deploy/sync_from_train.sh
git add scripts/deploy/sync_to_train.sh scripts/deploy/sync_from_train.sh
git commit -m "feat(deploy): bi-directional rsync scripts (DEV ↔ TRAIN)"
```

---

### Task 9.2: train_monitor.py + track_best_ckpt.py

**Files:**
- Create: `scripts/deploy/track_best_ckpt.py`
- Create: `scripts/deploy/train_monitor.py`

- [ ] **Step 1: Write track_best_ckpt.py**

```python
"""Maintain a `best_checkpoint` symlink on TRAIN pointing at the highest
ToMBench subset500 score across all checkpoints.

Runs on TRAIN (inside or alongside the train container). Reads tensorboard
event files for the `validation/tombench_subset500_accuracy` scalar.
"""
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def find_latest_event_dir(tb_root: Path) -> Path | None:
    candidates = sorted(tb_root.glob("**/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1].parent


def get_best_step(tb_dir: Path, tag: str) -> tuple[int, float] | None:
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    best = max(ea.Scalars(tag), key=lambda s: s.value)
    return best.step, best.value


def find_ckpt_for_step(ckpt_root: Path, step: int) -> Path | None:
    cand = list(ckpt_root.glob(f"checkpoint-{step}"))
    if cand:
        return cand[0]
    # ROLL may name checkpoints differently
    cand = list(ckpt_root.glob(f"*step{step}*"))
    return cand[0] if cand else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True,
                   help="dir containing checkpoint-N subdirs")
    p.add_argument("--tb-root", required=True,
                   help="dir containing TensorBoard event files")
    p.add_argument("--tag", default="validation/tombench_subset500_accuracy")
    p.add_argument("--best-symlink", default=None,
                   help="symlink path; default <ckpt-root>/best_checkpoint")
    p.add_argument("--loop", action="store_true", help="re-check every 60s")
    args = p.parse_args()

    ckpt_root = Path(args.ckpt_root)
    tb_root = Path(args.tb_root)
    sym = Path(args.best_symlink) if args.best_symlink else ckpt_root / "best_checkpoint"

    while True:
        tb_dir = find_latest_event_dir(tb_root)
        if tb_dir is None:
            print("waiting for tb events ...")
        else:
            res = get_best_step(tb_dir, args.tag)
            if res is None:
                print(f"tag {args.tag} not yet in tb")
            else:
                step, score = res
                ckpt = find_ckpt_for_step(ckpt_root, step)
                if ckpt is None:
                    print(f"best step {step} ({score:.4f}) but ckpt dir not found")
                else:
                    if sym.exists() or sym.is_symlink():
                        sym.unlink()
                    sym.symlink_to(ckpt.resolve())
                    print(f"best step {step} ({score:.4f}) → {sym} -> {ckpt}")
        if not args.loop:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write train_monitor.py (early-stop daemon)**

```python
"""External early-stop monitor for ROLL training.

Watches the latest tensorboard event file. If any termination condition is
met, the monitor calls `docker kill <container>` to terminate training.
Best-checkpoint symlinking continues regardless.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import time
from collections import deque
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def latest_tb_dir(tb_root: Path) -> Path | None:
    events = sorted(tb_root.glob("**/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    return events[-1].parent if events else None


def read_scalar(tb_dir: Path, tag: str):
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def should_stop(tb_dir: Path) -> tuple[bool, str]:
    """Return (should_stop, reason)."""
    # Rule 1: KL > 0.5 for 3 consecutive eval windows
    kl = read_scalar(tb_dir, "actor/kl")
    if len(kl) >= 3 and all(v > 0.5 for _, v in kl[-3:]):
        return True, f"kl>0.5 for 3 consecutive ({[v for _,v in kl[-3:]]})"

    # Rule 2: entropy < 0.1 for 3 consecutive
    ent = read_scalar(tb_dir, "actor/entropy")
    if len(ent) >= 3 and all(v < 0.1 for _, v in ent[-3:]):
        return True, f"entropy<0.1 for 3 consecutive ({[v for _,v in ent[-3:]]})"

    # Rule 3: subset500 acc continuously > 3pp below best
    acc = read_scalar(tb_dir, "validation/tombench_subset500_accuracy")
    if len(acc) >= 3:
        best = max(v for _, v in acc)
        recent = [v for _, v in acc[-3:]]
        if all(v < best - 0.03 for v in recent):
            return True, f"subset500 3 consecutive evals < best-0.03 (best={best:.4f}, recent={recent})"

    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb-root", required=True)
    p.add_argument("--container", required=True,
                   help="docker container name to kill on stop")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()

    tb_root = Path(args.tb_root)
    while True:
        tb_dir = latest_tb_dir(tb_root)
        if tb_dir is not None:
            stop, reason = should_stop(tb_dir)
            if stop:
                print(f"[monitor] EARLY STOP: {reason}")
                subprocess.run(["docker", "kill", args.container], check=False)
                return
            else:
                print(f"[monitor] OK at {time.strftime('%H:%M:%S')}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/deploy/track_best_ckpt.py scripts/deploy/train_monitor.py
git commit -m "feat(deploy): best-ckpt symlink + external early-stop monitor"
```

---

### Task 9.3: convert_megatron_to_hf.py

**Files:**
- Create: `scripts/deploy/convert_megatron_to_hf.py`

- [ ] **Step 1: Write conversion wrapper**

```python
"""Convert ROLL/Megatron checkpoint → HuggingFace format for vLLM serving.

ROLL ships `mcore_adapter` to handle this. This script is a thin CLI wrapper.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Megatron checkpoint dir")
    p.add_argument("--dst", required=True, help="HF-format output dir")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B",
                   help="reference HF model for tokenizer/config")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"src {src} missing")
    dst.mkdir(parents=True, exist_ok=True)

    # Use ROLL's mcore_adapter conversion script
    converter = Path("framework/ROLL/mcore_adapter/scripts/convert_to_hf.py")
    if converter.exists():
        print(f"[convert] using {converter}")
        subprocess.run([
            "python", str(converter),
            "--load", str(src),
            "--save", str(dst),
            "--base-model", args.base_model,
        ], check=True)
    else:
        # Fallback: if final ckpt is already HF (some ROLL strategies), just copy
        config = src / "config.json"
        if config.exists():
            print(f"[convert] {src} appears to be HF format already; copying")
            for item in src.iterdir():
                d = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, d)
        else:
            raise SystemExit(
                f"converter {converter} not found and {src} is not HF format. "
                "Install ROLL mcore_adapter or manually convert."
            )
    print(f"[convert] done → {dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/deploy/convert_megatron_to_hf.py
git commit -m "feat(deploy): Megatron→HF checkpoint conversion wrapper"
```

---

### Task 9.4: env_check_remote.sh

**Files:**
- Create: `scripts/deploy/env_check_remote.sh`

- [ ] **Step 1: Write remote env check**

```bash
#!/usr/bin/env bash
set -euo pipefail

source configs/deploy.env

echo "[remote-env-check] checking ${TRAIN_HOST} ..."
ssh -i "${TRAIN_SSH_KEY}" "${TRAIN_HOST}" bash -c "'
echo \"--- nvidia-smi ---\"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo \"--- docker ---\"
docker --version
docker compose version
echo \"--- disk ---\"
df -h \"${TRAIN_PATH}\" \"${TRAIN_OUTPUT_DIR}\" \"${TRAIN_DATA_DIR}\" \"${TRAIN_MODELS_DIR}\" 2>/dev/null || true
echo \"--- mounts ---\"
ls -la \"${TRAIN_PATH}\" 2>/dev/null | head -10 || echo \"(${TRAIN_PATH} not yet created)\"
'"
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/deploy/env_check_remote.sh
git add scripts/deploy/env_check_remote.sh
git commit -m "feat(deploy): remote environment check script"
```

---

## Phase 10 — Analysis scripts

### Task 10.1: plot_training_curves.py

**Files:**
- Create: `scripts/analysis/__init__.py` (empty)
- Create: `scripts/analysis/plot_training_curves.py`

- [ ] **Step 1: Write the plot script**

```python
"""Plot training curves from TensorBoard logs into a single PNG.

Reads output/tensorboard/ and produces output/analysis/curves.png.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


TAGS = [
    ("critic/rewards/mean",                    "Reward (mean)"),
    ("reward/r_fmt_mean",                      "R_fmt"),
    ("reward/r_out_mean",                      "R_out"),
    ("reward/r_len_mean",                      "R_len"),
    ("actor/loss",                             "Actor loss"),
    ("actor/kl",                               "KL"),
    ("actor/entropy",                          "Entropy"),
    ("actor/ppo_ratio_high_clipfrac",          "Clip high frac"),
    ("actor/ppo_ratio_low_clipfrac",           "Clip low frac"),
    ("response_length/mean",                   "Response len (mean)"),
    ("validation/tombench_subset500_accuracy", "Subset500 acc"),
    ("reward/r_total_mean",                    "R_total"),
]


def latest_event_dir(root: Path) -> Path | None:
    events = sorted(root.glob("**/events.out.tfevents.*"),
                    key=lambda p: p.stat().st_mtime)
    return events[-1].parent if events else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb", default="output/tensorboard")
    p.add_argument("--out", default="output/analysis/curves.png")
    args = p.parse_args()

    root = Path(args.tb)
    tb_dir = latest_event_dir(root)
    if tb_dir is None:
        print(f"No tensorboard events under {root}")
        return

    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])

    n = len(TAGS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for i, (tag, title) in enumerate(TAGS):
        ax = axes[i // cols][i % cols]
        if tag in available:
            scalars = ea.Scalars(tag)
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            ax.plot(steps, values)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
mkdir -p scripts/analysis
touch scripts/analysis/__init__.py
git add scripts/analysis/__init__.py scripts/analysis/plot_training_curves.py
git commit -m "feat(analysis): training curves plotter (12-panel PNG)"
```

---

### Task 10.2: diff_eval_results.py

**Files:**
- Create: `scripts/analysis/diff_eval_results.py`

- [ ] **Step 1: Write diff script**

```python
"""Compare two eval result JSONs (e.g., baseline vs trained) into markdown."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.eval.report import aggregate_results, format_markdown_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="output/eval/baseline_combined.json")
    p.add_argument("--trained", default="output/eval/final.json")
    p.add_argument("--out", default="output/analysis/eval_diff.md")
    args = p.parse_args()

    base_records = json.loads(Path(args.baseline).read_text()) if Path(args.baseline).exists() else []
    train_records = json.loads(Path(args.trained).read_text()) if Path(args.trained).exists() else []

    base_agg = aggregate_results(base_records)
    train_agg = aggregate_results(train_records)

    lines = ["# Eval diff: baseline vs trained", ""]

    # Combined main table
    combined: dict = {}
    for k, v in base_agg.items():
        combined[k] = v
    for k, v in train_agg.items():
        combined[k] = v
    lines.append(format_markdown_table(combined))

    # Compute deltas vs deepseek-v4-pro X
    direct_keys = [k for k in combined if k[1] == "direct"]
    x = next((combined[k]["overall"] for k in direct_keys if "deepseek" in k[0]), None)
    if x is not None:
        lines.append("")
        lines.append("## Distance to deepseek-v4-pro (X) on direct overall")
        lines.append("| Model | overall | X − overall | meets ε=0.02? |")
        lines.append("|---|---|---|---|")
        for k in sorted(direct_keys):
            overall = combined[k]["overall"]
            delta = x - overall
            status = "✓ (or better)" if delta <= 0.02 else "✗"
            lines.append(f"| {k[0]} | {overall:.4f} | {delta:+.4f} | {status} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/analysis/diff_eval_results.py
git commit -m "feat(analysis): baseline-vs-trained eval diff in markdown"
```

---

### Task 10.3: error_audit.py

**Files:**
- Create: `scripts/analysis/error_audit.py`

- [ ] **Step 1: Write error audit**

```python
"""Audit incorrect answers from the trained model: 5 sample errors per task."""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="output/eval/final.json")
    p.add_argument("--out", default="output/analysis/errors.md")
    p.add_argument("--per-task", type=int, default=5)
    args = p.parse_args()

    records = json.loads(Path(args.results).read_text())
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not r["correct"] and r.get("protocol") == "direct":
            by_task[r["task"]].append(r)

    lines = ["# Error audit (direct protocol, trained model)", ""]
    random.seed(42)
    for task, errs in sorted(by_task.items()):
        lines.append(f"## {task} — {len(errs)} errors total")
        lines.append("")
        sample = random.sample(errs, k=min(args.per_task, len(errs)))
        for r in sample:
            lines.append(f"### qid: {r['question_id']} ({r['language']})")
            lines.append(f"- gold: **{r['gold']}**, pred: **{r['pred']}**")
            resp = (r.get("raw_responses") or [""])[0]
            resp = resp[:500] + "..." if len(resp) > 500 else resp
            lines.append("- raw response:")
            lines.append(f"  ```\n  {resp}\n  ```")
            lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/analysis/error_audit.py
git commit -m "feat(analysis): per-task error audit (5 samples each)"
```

---

## Phase 11 — Documentation + final wire-up

### Task 11.1: Fill out runbook.md

**Files:**
- Modify: `docs/runbook.md`

- [ ] **Step 1: Replace placeholder with full runbook**

```markdown
# Runbook

Operational steps for each stage. See `docs/superpowers/specs/2026-05-11-qwen3-8b-tombench-rlvr-design.md` for design rationale.

## Prerequisites

1. Copy `configs/deploy.env.example` to `configs/deploy.env` and fill in TRAIN host/path/SSH key.
2. Export API keys in your shell (do NOT commit them):
   ```bash
   export DEEPSEEK_API_KEY=...
   export DASHSCOPE_API_KEY=...
   ```
3. Build the DEV docker image once: `docker compose -f docker/dev/docker-compose.yml build dev`

## Stage 0 — Environment check

```bash
make env-check
```
Expected: `ALL OK`. If anything fails, fix before proceeding.

Then verify the TRAIN host is reachable:
```bash
bash scripts/deploy/env_check_remote.sh
```

## Stage 1 — Build training data

```bash
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_tombench_eval.py
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_socialiqa.py
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_simpletom.py
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_exploretom.py
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_hitom.py
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/synth_tomtype.py --n 1500
make build-data
```
**Go/no-go:**
- `data/tom/tom_train.jsonl` has ≥ 7000 records
- `data/tom/dedup_report.json` shows all `per_source_max_jaccard_distribution.*.max ≤ 0.6`

## Stage 2 — Baseline measurement

```bash
make baseline
```
Records `Y_base_nt`, `Y_base_t`, `X` in `output/eval/baseline_report.md`.

## Stage 3 — Reward worker unit tests

```bash
make test-reward
make test-eval
make test-data
```
All must pass.

## Stage 4 — Stage-1 training (small-scale verification)

On DEV:
```bash
make sync-up
```
Then on TRAIN (via DEV):
```bash
make train-stage1
```
This runs in foreground. To run detached:
```bash
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" \
  "cd $TRAIN_PATH && docker compose -f docker/train/docker-compose.yml \
   --env-file configs/deploy.env up -d train"
```

In another terminal on TRAIN, start the early-stop monitor and best-ckpt tracker:
```bash
ssh -i "$TRAIN_SSH_KEY" "$TRAIN_HOST" \
  "cd $TRAIN_PATH && \
   python scripts/deploy/track_best_ckpt.py \
     --ckpt-root $TRAIN_OUTPUT_DIR/qwen3-8B-tombench-rlvr-stage1 \
     --tb-root $TRAIN_OUTPUT_DIR/tensorboard/qwen3-8B-tombench-rlvr-stage1 \
     --loop &
   python scripts/deploy/train_monitor.py \
     --tb-root $TRAIN_OUTPUT_DIR/tensorboard/qwen3-8B-tombench-rlvr-stage1 \
     --container <train_container_name> &"
```

When done, pull back to DEV:
```bash
make sync-down
make analyze
```
**Go/no-go (after 200 steps):**
- `reward/r_fmt_mean` > 0.95
- `reward/r_out_mean` > 0.8 × `Y_base_nt`
- `reward/r_len_mean` > 0.7
- Subset500 acc > `Y_base_nt`

## Stage 5 — Stage-2 main training

```bash
make pipeline-stage2
```
Runs ~25 hours. Same monitoring as stage-1.

## Stage 6 — Final evaluation

After `make pipeline-stage2` completes, `make serve-launch && make eval-final` run automatically as part of the pipeline. To re-run manually:
```bash
make serve-launch
make eval-final
make analyze
```

**Final judgement:**
- `Y'_direct ≥ X` → "surpasses" deepseek-v4-pro
- `X − 0.02 ≤ Y'_direct < X` → "approaches"
- `Y'_direct < X − 0.02` → triggers L3 fallback (stage 7)

## Stage 7 — L3 fallback (only if stage 6 misses target)

L3 requires building a process-reward model. See `configs/tombench-rlvr/rlvr_config_stage3_l3.yaml` (skeleton) and implement the additional reward worker before running `make pipeline-l3`.

## Common failure modes

| Symptom | Action |
|---|---|
| `make build-data` complains some source is empty | OK as long as `tom_train.jsonl` ≥ 7000; other sources compensated |
| Stage-4 health checks fail | Re-tune one config knob per the table in spec §7.7; re-run stage-1 |
| `actor/kl > 0.5` warning | Lower `learning_rate` to 5e-7 in config; resume from latest ckpt |
| OOM during training | Reduce `gradient_accumulation_steps` from 32→16 or set `recompute_granularity: full` |
| Subset500 acc keeps falling | Early-stop will kill training; use whatever best ckpt was saved |
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbook.md
git commit -m "docs: full operational runbook for all 7 stages"
```

---

### Task 11.2: Fill out eval-protocol.md

**Files:**
- Modify: `docs/eval-protocol.md`

- [ ] **Step 1: Write protocol document**

```markdown
# Evaluation Protocols

Three protocols. Main score for "approaches/surpasses deepseek-v4-pro" is **protocol 1 (Direct)**.

## Protocol 1 — Direct answer (MAIN SCORE)

**System prompt:**
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

**User prompt (English):**
```
Story:
{STORY}

Question: {QUESTION}
A. {OPTION-A}
B. {OPTION-B}
C. {OPTION-C}
D. {OPTION-D}
```

**User prompt (Chinese):** same shape, Chinese labels `故事:`, `问题:`.

**Sampling:** `temperature=0.0, top_p=1.0, max_tokens=32`

**Answer extraction:** `\boxed{[A-D]}` first match; fallback to first capital letter A-D in response.

**Why this is the main protocol:** training is done with this exact prompt + extraction; train/test consistency is critical.

## Protocol 2 — CoT (reference score)

**System prompt:** allows step-by-step thinking, requires final `\boxed{X}` on the last line.

**Sampling:** `temperature=0.6, top_p=0.9, max_tokens=1024`

**Answer extraction:** last `\boxed{[A-D]}`; fallback to last capital letter in tail 200 chars.

## Protocol 3 — DEL-ToM (optional)

For belief-class subtasks only (False Belief, Unexpected Outcome, Knowledge). Generates **N=8 CoT samples** at `temperature=0.7`; majority vote, alphabetic tiebreak.

Not used during training. Only for final evaluation enhancement.

## Reporting

For each model:
- Overall accuracy (all 2860 questions × 2 languages = ~5720 records)
- Per-language: EN, ZH
- Per-task (8 ToMBench broad tasks): False Belief, Strange Story, Unexpected Outcome, Persuasion Story, Knowledge, Desire, Emotion, Intention, Non-literal Comm
```

- [ ] **Step 2: Commit**

```bash
git add docs/eval-protocol.md
git commit -m "docs: evaluation protocol document (direct/cot/del-tom)"
```

---

### Task 11.3: Add convenience Makefile target for serve url printout

**Files:**
- Modify: `Makefile` (already exists from Task 0.4)

- [ ] **Step 1: Add `serve-url` target after `serve-launch`**

In `Makefile`, after the `serve-launch` target, insert:

```makefile
serve-url: ## Print the OpenAI-compatible base_url to use for eval
	@echo "Base URL: http://$(TRAIN_HOST_HOSTONLY):$(SERVE_PORT)/v1"
	@echo "Model id: qwen3-8b-tom"
	@echo
	@echo "Test:"
	@echo "  curl http://$(TRAIN_HOST_HOSTONLY):$(SERVE_PORT)/v1/models"

TRAIN_HOST_HOSTONLY = $(shell echo "$(TRAIN_HOST)" | sed 's/.*@//')
```

- [ ] **Step 2: Verify**

```bash
cd /Users/jaredguo-mini/develop/training
make help | grep serve-url
```

Expected: shows `serve-url            Print the OpenAI-compatible base_url to use for eval`

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add serve-url target for inspecting deployed model endpoint"
```

---

## Self-Review

After writing the plan, I reviewed it against the spec:

**Spec coverage check:**
- §1 Goal & success criteria → Task 5.1 produces `Y_base_nt/Y_base_t/X`, Task 11.1 documents the judgement rule ✓
- §2 Training data → Tasks 3.1, 4.1-4.6 ✓
- §3 Evaluation protocols → Tasks 2.1-2.4 (extractors, clients, run_tombench, report) + Task 11.2 (eval-protocol.md) ✓
- §4 System prompt + user templates → Task 2.4 includes `SYSTEM_PROMPT_DIRECT` + `build_user_prompt_zh/en` ✓
- §5 L2 reward → Task 6.1 (`TomMcqRewardWorker`) ✓
- §5 L3 fallback → Task 7.3 (config skeleton, full implementation deferred until triggered) ✓
- §6 Training configs → Tasks 7.1, 7.2 ✓
- §7 Cross-machine + Docker + Makefile → Tasks 1.1-1.2 (dev), 8.1 (train), 8.2 (serve), 0.4 (Makefile), 9.1-9.4 (deploy) ✓
- §8 Risks + monitoring + early-stop → Task 9.2 (monitor + best-ckpt) ✓
- §9 Deliverables → all files created across tasks ✓

**Placeholder scan:** No "TBD/TODO/implement later" patterns in any task body. The only deliberate placeholder is `configs/tombench-rlvr/rlvr_config_stage3_l3.yaml` which is documented in the file as a skeleton activated only on stage-2 miss.

**Type consistency:** Verified `TomRecord` dataclass fields are consistently named across `schema.py`, `build_*.py`, and `merge_and_dedupe.py` (question_id, source, language, task, story, question, opt_a/b/c/d, gold). Reward function signature `tom_mcq_reward_fn(response, response_token_count, ground_truth, l_min, l_max, k)` matches what `compute_rewards` calls. Extractor function names (`extract_direct`, `extract_cot`, `vote_del_tom`) match between `extractors.py`, `run_tombench.py`, and the tests.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-12-qwen3-8b-tombench-rlvr.md`.**
