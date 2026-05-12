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
