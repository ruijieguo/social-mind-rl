# Qwen3-14B Stage 22 — "Plan A" (unshackle the thinking budget)

Self-contained training + per-ckpt eval for **Stage 22 / v4.0**, the first 14B run
designed to **beat base on the 4-benchmark mean** instead of overfitting ToMBench.

## Why (the 2026-05-30 findings)

The full eval (`experiment/qwen3-14b-full-eval/`) proved the 14B RLVR line traded
general ToM ability for ToMBench points — best-protocol 4-bench mean: **base 0.7603 >
v3.5 0.7389 > v3.1 0.7305**. Three root causes, all fixed here:

1. **Thinking compressed 4-5×** by `response_length=256` + a **multiplicative** reward
   whose `r_len` zeroes any answer thinking >`l_max=256` tokens → Hi-ToM collapses
   (regression monotonic in reasoning depth). Docs: `docs/insight_and_optimization_2026-05-30.md`.
2. **26% of training data silently dropped** — ExploreToM + higher-order "HOT" synth
   were `tag=None` → routed to domain `math_rule` → filtered out. Docs: `docs/data_audit_2026-05-30.md`.
3. **gold A-bias** (A=45%) miscalibrating the answer prior.
   Model soup (`docs/...`) confirmed weight interpolation can't beat base → must retrain.

## What Plan A does

| Lever | v3.1 | Plan A |
|---|---|---|
| `response_length` | 256 | **2048** |
| reward aggregation | multiplicative | **weighted_sum** (length = 5% soft pref) |
| reward `l_max` | 256 | **2048** (`l_max_long` 4096) |
| KL anchor | off | **`use_kl_loss` + `kl_loss_coef` 0.001 → BASE** |
| init / reference | stage12 | **base Qwen3-14B** |
| data | dropped 26% + A-bias | **fixed** (re-tagged + gold ~25% each) |
| ckpt selection | ToMBench | **4-bench mean** (offline, per ckpt) |

**Success = 4-bench mean > 0.7603, with Hi-ToM cot ≥ 0.76 (proof the枷锁 is gone).**
Design + risks: `docs/planA_design_2026-05-30.md`.

## ⚠️ Reward params live in JSON, not YAML

ROLL's `RewardConfig` dataclass **silently drops** `l_max`/`aggregation`/`r_*_weight`
from the YAML rewards block (memory: `roll_rewardconfig_drops_custom_keys`). The
authoritative values are in **`configs/tom_reward_planA.json`**, loaded by the worker
via `TOM_REWARD_OVERRIDE` (the train entrypoint sets it automatically for
`*_qwen3_14b*` stages). The worker logs the resolved params —
**VERIFY at launch**: `grep '[tom_mcq_reward] resolved' <train log>` should show
`l_max=2048.0 aggregation=weighted_sum`. The override mechanism (`apply_reward_override`)
was ported into `framework/ROLL/.../tom_mcq_reward_worker.py` (tested).

## Layout

```
experiment/qwen3-14b/
├── configs/
│   ├── rlvr_config_stage22_qwen3_14b.yaml   # the Plan A RLVR config
│   ├── tom_reward_planA.json                # AUTHORITATIVE reward params (override)
│   └── deploy.env.191                       # .191 train + eval paths
├── scripts/
│   ├── data/build_planA_data.py             # builds the fixed training data
│   ├── parallel_eval.py, prompts.py         # 4-bench eval engine
│   ├── 0{1,2,4,6}_*.sh, 05_aggregate_report.py, soup_summary.py
├── docker/docker-compose.yml                # serve (per-ckpt eval)
└── docs/                                     # insight / planA design / data audit
```

## Run (on .191)

```bash
# 0. Prereqs on .191 (sync from .181 / DEV):
#    /home/h800/grj-projects/models/Qwen3-14B            (base, 8 shards)
#    /home/h800/grj-projects/tom-data/tom_train_stage22_planA.jsonl
#    /home/h800/grj-projects/tom-data/tombench_eval_subset500.jsonl
#    (regenerate data: python scripts/data/build_planA_data.py \
#       --in .../tom_train_stage14_weighted.jsonl --out .../tom_train_stage22_planA.jsonl)

# 1. Train (project docker/train compose; entrypoint routes STAGE → this dir)
STAGE=stage22_qwen3_14b \
TRAIN_DATA_DIR=/home/h800/grj-projects/tom-data \
TRAIN_MODELS_DIR=/home/h800/grj-projects/models \
TRAIN_OUTPUT_DIR=/home/h800/grj-projects/tom-output \
docker compose -f docker/train/docker-compose.yml up | tee logs/train_stage22.log

# 2. VERIFY reward override took effect
grep '\[tom_mcq_reward\] resolved' logs/train_stage22.log   # expect l_max=2048 weighted_sum

# 3. Per ckpt: convert Megatron→HF, then 4-bench eval, pick best by mean
#    (reuse the full-eval harness: serve → parallel_eval.py → soup_summary.py)
```

## Watch during training

- `r_out_mean` (correctness) and **rollout mean response length** — should be **>256**
  (proof the budget is unshackled; if it stays ~256 the override didn't take effect).
- in-loop val is ToMBench-only → **do offline 4-bench eval per ckpt** for real selection.
- **late collapse** (cf. stage17 Dr.GRPO step-100 collapse) — `save_steps=25` + watch val.
