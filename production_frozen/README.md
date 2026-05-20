# Production Frozen — Reproducibility Snapshot

> **Status**: Best-of-project models pinned. Do NOT modify files in this directory.
> **Created**: 2026-05-20 (after Stage 10 abort confirmed Stage 7/8 are project-best)

## Headline results (frozen)

| Model | Full 5718 direct | Clean 4551 direct | Subset500 best | HF model size |
|---|---|---|---|---|
| **qwen3-8b-tom-stage7** | **0.7419** | 0.8321 | cot 0.7460 | 16 GB |
| **qwen3-14b-tom-stage8** | **0.7594** | 0.8449 | del_tom **0.7920** | 28 GB |
| deepseek-v4-pro (reference) | 0.8080 | 0.9013 | direct 0.7880 | API |
| GPT-5.5 (reference) | 0.8349 | 0.9343 | — | API |

**Stage 8 14B subset500 del_tom 0.7920 反超 deepseek-v4-pro direct 0.7880 (+0.4pp)** — 项目最大成就。

## Directory contents

```
production_frozen/
├── README.md                          # this file
├── SHA256SUMS.txt                     # cryptographic checksums of every artifact
├── configs/
│   ├── rlvr_config_8b_stage7_FROZEN.yaml      # exact config that produced 8B stage7
│   └── rlvr_config_14b_stage8_FROZEN.yaml     # exact config that produced 14B stage8
├── data/
│   ├── tom_train_8b_stage7.jsonl              # 9559 records (= tom_train.jsonl at time of stage7)
│   ├── tom_train_14b_stage8.jsonl             # 9259 records (= tom_train.jsonl at time of stage8)
│   ├── tombench_eval.jsonl                    # 5718 questions (full ToMBench eval)
│   ├── tombench_eval_subset500.jsonl          # 500 deterministic subset
│   ├── tombench_eval_clean.jsonl              # 4551 (after GPT-5.5 audit)
│   ├── clean_eval_qids.json                   # GPT-5.5 audit keep/drop list
│   └── raw/
│       ├── synth_gpt55.jsonl                  # Stage 6 base synth (1400)
│       ├── synth_gpt55_phase_a.jsonl          # Phase A.1 (1500, used in 8B stage7)
│       ├── synth_gpt55_phase_b_zh.jsonl       # Phase A.2 (800 ZH, used in both)
│       └── synth_gpt55_phase_c.jsonl          # Phase C (1200, used in 14B stage8)
├── scripts/
│   ├── tom_mcq_reward_worker.py               # frozen reward worker code
│   └── run_tombench.py                        # frozen eval framework
├── logs/
│   ├── train_8b_stage7.log                    # full training log (12 MB)
│   └── train_14b_stage8.log                   # full training log (17 MB)
└── eval/
    ├── 8b_stage7_full5718.json
    ├── 8b_stage7_clean_eval.json
    ├── 8b_stage7_subset500.json
    ├── 14b_stage8_full5718.json
    ├── 14b_stage8_clean_eval.json
    ├── 14b_stage8_subset500.json
    ├── deepseek_full5718.json
    └── gpt-5.5_full5718.json
```

## Reproducibility verification

To verify nothing has been corrupted:

```bash
cd production_frozen
shasum -a 256 -c <(grep -E "^[0-9a-f]{64}" SHA256SUMS.txt)
```

Expected output: `OK` for every line.

## Trained model checkpoints (TRAIN host)

The checkpoints themselves are too large for git (~196 GB Megatron + 28 GB HF for 14B).
They live on the TRAIN host at:

```
TRAIN host: h800@172.16.120.181 (configurable via configs/deploy.env)

8B Stage 7:
  Megatron: /data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage7-1x8/20260518-192042/checkpoint-249/
  HF:       /data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage7/  (16 GB)

14B Stage 8:
  Megatron: /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage8-1x8/20260518-234128/checkpoint-349/
  HF:       /data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf-stage8/  (28 GB)
```

To preserve the HF models off-cluster, run:

```bash
# From DEV machine
mkdir -p /path/to/archive
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  h800@172.16.120.181:/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage7/ \
  /path/to/archive/qwen3-8B-tom-hf-stage7/

rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  h800@172.16.120.181:/data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf-stage8/ \
  /path/to/archive/qwen3-14B-tom-hf-stage8/
```

## Reproducing the training from scratch

### Prerequisites
- 1× H800 node with 8 GPUs (80 GB each)
- 14 TB NVMe at `/data_nvme/`
- Docker images: `qwen3-tom-train:latest` (built from `docker/train/Dockerfile`)

### 8B Stage 7 (raw 0.7419)

```bash
# 1. Place data
cp production_frozen/data/tom_train_8b_stage7.jsonl /mnt/data/tom_train.jsonl
cp production_frozen/data/tombench_eval_subset500.jsonl /mnt/data/

# 2. Place config
cp production_frozen/configs/rlvr_config_8b_stage7_FROZEN.yaml \
   configs/tombench-rlvr/rlvr_config_8b_stage7_FROZEN.yaml

# 3. Train (via host shell, ~4.5h on 1×8 H800)
docker compose -f docker/train/docker-compose.yml \
  --env-file configs/deploy.env \
  run --rm --build -e STAGE=8b_stage7_FROZEN train

# 4. Convert Megatron → HF
docker run --rm --gpus all --ipc host --shm-size 8gb \
  --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
  -v $(pwd):/workspace \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  -e PYTHONPATH=/workspace/framework/ROLL/mcore_adapter/src \
  -w /workspace --entrypoint python qwen3-tom-train:latest \
  framework/ROLL/mcore_adapter/tools/convert.py \
  --checkpoint_path /mnt/output/qwen3-8B-tombench-rlvr-stage7-1x8/<timestamp>/checkpoint-249 \
  --output_path /mnt/output/qwen3-8B-tom-hf-stage7 --bf16
```

### 14B Stage 8 (raw 0.7594)

```bash
# 1. Place data (note different file)
cp production_frozen/data/tom_train_14b_stage8.jsonl /mnt/data/tom_train.jsonl

# 2-4. Same as 8B but with 14B config and longer training (~7h on 1×8 H800)
docker compose ... -e STAGE=14b_stage8_FROZEN train
```

## Reproducing eval

```bash
# Start vLLM serve
docker run --rm -d --name eval-serve \
  --gpus device=0 --ipc host --shm-size 16gb -p 8000:8000 \
  -v /mnt/output:/mnt/output \
  --entrypoint python qwen3-tom-train:latest \
  -m vllm.entrypoints.openai.api_server \
  --model /mnt/output/qwen3-14B-tom-hf-stage8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --served-model-name qwen3-14b-tom-stage8

# From DEV
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy dev \
  python production_frozen/scripts/run_tombench.py \
    --backend openai \
    --base-url http://172.16.120.181:8000/v1 \
    --model qwen3-14b-tom-stage8 \
    --data production_frozen/data/tombench_eval.jsonl \
    --protocols direct --concurrency 32 \
    --output reproduce_stage8_full5718.json
```

Expected: full 5718 direct = **0.7594** (±0.001 due to vLLM determinism).

## Critical reproducibility constraints

### Reward worker behavior
The frozen `tom_mcq_reward_worker.py` adds a `aggregation` parameter that did NOT
exist when stage 7/8 trained, but its **default is `"multiplicative"`** which
matches the original behavior. The frozen configs do NOT set `aggregation:` so
they fall back to the default. If you want bit-exact reproduction:

```yaml
# Optional: explicitly pin to original behavior
rewards:
  tom_mcq:
    aggregation: multiplicative   # this was the only behavior at stage 7/8 train time
```

### Algorithmic invariants (DO NOT change for reproduction)
- `whiten_advantages: true`
- `add_token_level_kl: false`
- `use_kl_loss` not set (defaults to false)
- `entropy_loss_coef` not set (defaults to 0)
- `loss_agg_mode: "seq-mean-token-mean"` (default)
- `pg_clip_low: 0.20, pg_clip_high: 0.28` (DAPO Clip-Higher)
- `dual_clip_loss: true`
- `difficulty_low_threshold: 0.1, high_threshold: 0.95`
- `response_length: 256`
- `prompt_length`: 2048 for 8B, 1024 for 14B (different — KV cache constraint)
- `pretrain`: `Qwen/Qwen3-8B` for 8B, `Qwen/Qwen3-14B` for 14B (NO SFT init)

### Data invariants
- 8B stage7 trained on **9559 records** (= base 7259 + Phase A.1 1500 + Phase A.2 800)
- 14B stage8 trained on **9259 records** (= base 7259 + Phase B.2 800 + Phase C 1200)
- Both use the same 7259 cleaned base (post GPT-5.5 audit drop of ExploreToM/simpletom_zh)
- Eval sets: 5718 (full) + 500 (subset500) + 4551 (clean, post GPT-5.5 audit)

### Hardware invariants
- 1×8 H800 80GB SXM (NVLink full mesh)
- 80 GB GPU memory required for stage 8 14B with TP=2
- Distributed optimizer save uses Gloo+CPU (`distrib_optim_fully_reshardable_mem_efficient: true`)
  to avoid OOM during checkpoint save

## What is NOT included (and why)

1. **Megatron checkpoints** (`iter_0000001/`): too large for git (~196 GB), live on TRAIN host
2. **Distributed optimizer state**: ~96 GB, only useful if resuming training, skip for inference
3. **HF safetensors**: 16-28 GB, would bloat repo. Pull from TRAIN host as needed.
4. **Qwen/Qwen3-14B base weights**: download from ModelScope/HuggingFace at training time

## Failed experiments NOT in this snapshot

These training stages produced models but were **worse** than stage 7/8 — they are
preserved in repo history and `docs/` but not in production_frozen:

- Stage 9 (SFT cold start + KL + long CoT): 14B raw 0.7429, **-1.51pp vs s8**
- Stage 10 (weighted_sum reward + entropy): aborted at step 214, val 0.666 < s8 0.706
- Stage 1-6: all weaker than 7/8

See `docs/stage9_retro.md`, `docs/stage10_plan_evidence_based.md`, `docs/final_project_report.md`.

## Git tag for this frozen snapshot

```bash
git tag -a v1.0-production -m "Frozen at 8B stage7 + 14B stage8 best results"
git push origin v1.0-production
```

To check out this exact state:

```bash
git checkout v1.0-production -- production_frozen/
```

最后更新: 2026-05-20 14:30
