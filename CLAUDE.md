# CLAUDE.md — onboarding guide for agents + humans

## One-line summary

This repo trains **Qwen3-8B** with **GRPO** (on the ROLL framework) to maximize
accuracy on the **ToMBench** theory-of-mind benchmark, with **deepseek-v4-pro**
as the reference closed-source target.

Design spec: `docs/superpowers/specs/2026-05-11-qwen3-8b-tombench-rlvr-design.md`
Implementation plan: `docs/superpowers/plans/2026-05-12-qwen3-8b-tombench-rlvr.md`
Runbook: `docs/runbook.md` | Data card: `docs/data-card.md` | Eval protocol: `docs/eval-protocol.md`

## Workflow

**DEV (macOS)**: all data construction, eval-via-API, and analysis. Nothing
GPU-bound. One docker image (`qwen3-tom-dev:latest`).

**TRAIN (16×H800 Linux, TBD)**: Megatron+vLLM via ROLL RLVR pipeline. Two
docker images (`qwen3-tom-train`, `qwen3-tom-serve`).

**Transfer**: SSH + rsync via `make sync-up` / `make sync-down`. TRAIN host
is configured in `configs/deploy.env` (copy from `.example`).

Top-level entry: `make help`.

## Layout cheat-sheet

| Path | Purpose |
|---|---|
| `framework/ROLL/` | Vendored Alibaba ROLL framework (we only add one reward worker). `.git` was removed — treat as read-only upstream except for `roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` and `tests/test_tom_mcq_reward.py`. |
| `scripts/data/` | Build + dedupe + synth + translate training data. Tests under `scripts/data/tests/`. |
| `scripts/eval/` | OpenAI-compatible eval framework: clients, extractors, report, run_tombench, rebuild_baseline_report. Tests under `scripts/eval/tests/`. |
| `scripts/analysis/` | Post-hoc analysis: training curves, eval diff, error audit, baseline gap analysis. |
| `scripts/deploy/` | Bash/Python cross-machine orchestration: sync_to_train, train_monitor, best-ckpt tracker, converter. |
| `configs/tombench-rlvr/` | ROLL hydra configs for stage1 (small-scale), stage2 (main), stage3-l3 (fallback skeleton). |
| `docker/{dev,train,serve}/` | Three docker images + compose files. Only `dev` is buildable on macOS. |
| `data/tom/` (gitignored) | Training + eval JSONLs. `tom_train.jsonl` is the 5911-record merged set. |
| `output/eval/` (gitignored except baseline_report.md) | Evaluation results. |

## Data: known good state

- `tom_train.jsonl`: **5911 records** = ExploreToM 2000 + SimpleToM 1000 + synth 2911.
  Gold letter distribution ≈ 25% each after `rebalance_synth.py`.
- `tombench_eval.jsonl`: **5718 records** (2860 ToMBench × 2 languages, minus rows
  missing one language).
- `tombench_eval_subset500.jsonl`: random 500 for training-loop validation.
- **SocialIQa / Hi-ToM are NOT in the current data** — HF API changes made the
  dataset scripts unloadable; `synth_tomtype.py` generated 2911 records to
  compensate.
- Zero leakage against ToMBench (MinHash 4-gram Jaccard threshold 0.6, per
  `data/tom/dedup_report.json`).

## Baseline: known good numbers (on 500-question subset)

| Model | direct (main) | cot |
|---|---|---|
| **deepseek-v4-pro** = **target X** | **0.7880** | 0.7140 |
| **Qwen3-8B (non-thinking)** = **Y_base_nt (start)** | **0.6900** | 0.7640 |
| Qwen3-8B (thinking) = Y_base_t | 0.6940 | 0.7540 |

Report: `output/eval/baseline_report.md`.
- Approaches X: `Y'_direct ≥ 0.7680` (X − 0.02)
- Surpasses X: `Y'_direct ≥ 0.7880`

Per-task gap analysis (`scripts/analysis/baseline_gap_analysis.py`): the
highest-ROI training targets are False Belief (gap 20.8%, n=130) and
Emotion (gap 19.8%, n=86). Knowledge has a 34% both-wrong rate so training
upside is capped.

## API gotchas (non-obvious)

- **DashScope `qwen3-8b` + `enable_thinking=true` REQUIRES stream mode.**
  `ChatClient` auto-detects and switches. Do not remove.
- **Reasoning models (deepseek-v4-pro) burn `max_tokens` on hidden thinking.**
  The `direct` protocol uses `max_tokens=2048` specifically to accommodate
  reasoning models, even though non-reasoning models will return after a few
  tokens. Lowering this breaks deepseek-v4-pro baseline.
- **DashScope rate-limits tend to cap at ~1-2 req/s for personal keys.**
  Client backs off 30-60s on HTTP 429. Don't raise concurrency above 4.
- **DeepSeek-v4-flash** is ~2× faster than `-pro` for synthesis (no visible
  reasoning tokens). Use `-flash` for translation / synth; `-pro` for eval.

## Contributor rules

- Never run `make build-data` without first `make baseline` — the build
  overwrites `tom_train_4k.jsonl` and `tombench_eval_subset500.jsonl`, which
  baseline depends on. (Subset is seeded so it's reproducible.)
- Never raise `response_length` in stage1/2 configs above 256 without
  updating the `l_max` parameter in `TomMcqRewardWorker` config (and
  understanding that the spec explicitly chose 256 to penalize verbose
  thinking in this task).
- Never delete the ROLL vendored tree (`framework/ROLL/`). It's untracked by
  `.gitignore` except for our two custom files. Cloning upstream ROLL will
  restore the rest.
- Use `make test-data` / `make test-eval` / `make test-reward` before every
  commit. Currently: 52 unit tests should pass.

## Known TODOs

- Full 5718-question baseline is in progress (background task). Subset500
  results are representative but full-set will be authoritative for the
  final paper/report.
- Chinese training data was added (~2000 records via `translate_to_zh.py`)
  — after it finishes, rerun `merge_and_dedupe.py` to fold it into
  `tom_train.jsonl`.
- TRAIN host is not yet configured; `configs/deploy.env.example` → create
  the real `configs/deploy.env` to enable `make sync-up` / `make train-stage1`.
- L3 fallback (stage3) is a stub; implementation only if stage2 misses X−ε.

## When in doubt

- Check `docs/runbook.md` for command recipes.
- Check `docs/superpowers/specs/...` for design rationale.
- Check `docs/superpowers/plans/...` for the detailed task breakdown that
  built this.

Last updated: 2026-05-13.
