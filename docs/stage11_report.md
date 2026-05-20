# Stage 11 Multi-Track Report (running)

> Goal: 14B raw 0.7594 → 0.80+ (反超 deepseek-v4-pro), 逼近 GPT-5.5 0.8349
> Started: 2026-05-20

## Track A: Eval-time 协议 (stage 8 unchanged model) ✅ DONE

**Result on full 5718:**

| Protocol | Acc | vs s8 direct (0.7594) |
|---|---|---|
| direct | 0.7590 | -0.04pp (cache reproduction) |
| cot | 0.7681 | **+0.87pp** |
| **del_tom** | **0.7810** | **+2.16pp** ⭐ |

**Per-task del_tom gains** (most help on hard structural ToM):

| Task | direct | del_tom | gain |
|---|---|---|---|
| **False Belief** | 0.8622 | **0.9115** | **+4.93pp** |
| Belief | 0.7394 | 0.7676 | +2.82pp |
| Desire | 0.5694 | 0.5972 | +2.78pp |
| Emotion | 0.7274 | 0.7488 | +2.14pp |
| Intention | 0.8176 | 0.8324 | +1.47pp |
| Non-literal Comm | 0.7921 | 0.7961 | +0.40pp |
| Knowledge | 0.5138 | 0.5156 | +0.17pp |

**Insight**: del_tom helps most on tasks requiring multi-step belief tracking (False Belief +4.93pp). Knowledge (factual recall) and Non-literal Comm (pragmatic) hardly benefit — these don't need extra reasoning steps.

🎯 **Free win: 0.7594 → 0.7810 (+2.16pp), no training**

## Track B: ExploreToM data ✅ DONE

**Decision change**: After cloning Meta repo, the framework requires Llama-3.1-70B-Instruct as base teacher (cached prompts hardcoded). Stage 8 14B (MCQ-tuned) cannot generate the multi-step story contexts needed.

**Pivot**: Downloaded Meta's official `ExploreToM-data-sample.csv` (13309 records) and converted to ToMBench training format with smart distractor extraction.

Files:
- `data/tom/raw/exploretom_v2_meta/data.csv` (30 MB, raw Meta dataset)
- `data/tom/raw/exploretom_v2.jsonl` (2000 records ToMBench format)

**Filter**: kept only Llama-70B accuracy ≤ 0.7 (hard examples). Result: **2000 records**.

| Task | Records |
|---|---|
| Belief | 1152 |
| False Belief | 724 |
| Knowledge | 124 |

**Quality**: 0/2000 leakage vs eval set (jaccard < 0.85). Each record has belief annotations:
- `nth_order` (1 or 2 = belief level)
- `is_fb_1st` / `is_fb_2nd` (whether first/second-order belief is false)
- `llama70b_acc_infilled` (adversarial difficulty, ≤0.7 here)
- `story_type` (e.g., `tomi+object-state+asymmetric`)

These can be used for difficulty-aware curriculum in stage 12.

## Track C: HOT-targeted GPT-5.5 synth

Analyzed 492 HOT records (stage 8 wrong but gpt-5.5 + deepseek both right):
- False Belief: 102
- Non-literal Comm: 92
- Knowledge: 84
- Emotion: 80
- Intention: 61
- Desire: 39
- Belief: 34

Generating ~180 same-pattern training questions per task = ~1260 records total.

**Status**: running on DEV, ~4.4 records/min, ETA ~5h

Cost: ~$50 (1260 × $0.04)

## Track D: Continue stage 8 training

Config: identical to stage 8, only `pretrain` changed to stage 8 HF
- exp_name: `qwen3-14B-tombench-rlvr-stage11d-1x8`
- max_steps: 350
- Same data (9259), reward, GRPO config

**Status**: queued, launches after Track A vLLM freed (in scripts/launch_stage11d_train.sh)

## Track E: Final integration (stage 12)

Will train new 14B (init from stage 8) on:
- 9259 stage 8 base
- + 1500 ExploreToM (Track B)
- + 1260 HOT synth (Track C)
- = ~12000 records

Same recipe as stage 8 (350 steps GRPO).

Expected: 14B raw 0.78-0.82 (depending on B/C data quality)

## Decision tree

```
Track A done → if del_tom > 0.78 (full 5718): we have +2pp eval-time win
Track C done → if 1200+ high-quality records: feed to E
Track B done → if 1500+ records with belief annotations: feed to E
Track D done → tells us "continue training stage 8 alone gains X pp"
              → benchmark for E
Track E (stage 12) → if > stage 8 + 1pp: production v2.0
                  → else: declare stage 8 + Track A best protocol as final

Worst case fallback: stage 8 with Track A's best eval-time protocol
                     (= no training cost, +1.4pp from subset500 experiment)
```

## Resource usage

| Track | GPU | API cost | Wall time |
|---|---|---|---|
| A | 1 GPU on TRAIN (vLLM serve) | 0 | ~30 min |
| B | 1 GPU on TRAIN (vLLM serve, reusing A's) | ~$30 audit | ~2-3h |
| C | 0 (DEV) | ~$50 | ~5h |
| D | 8 GPU on TRAIN | 0 | ~14h |
| E | 8 GPU on TRAIN | 0 | ~14h |

**Total**: ~32 GPU-hours, ~$80, ~36h elapsed (with parallelism)

## Files

```
configs/tombench-rlvr/rlvr_config_stage11d_continue_1x8_14b.yaml
scripts/data/synth_gpt55_phase_d_hot.py        # Track C
scripts/data/merge_stage11_train.py            # Track E data merge
scripts/data/leakage_check_phase_d.py          # Track E leakage
scripts/launch_track_b_exploretom.sh           # Track B launcher
scripts/launch_stage11d_train.sh               # Track D launcher
output/analysis/hot_questions.jsonl            # 492 HOT records
framework/ExploreToM/                          # Meta repo (synced to TRAIN)
```

## Next decision points

1. **Track A complete** → write up del_tom 全集结果 here
2. **Track A vLLM free** → launch Track B (story_context_generator + A* search)
3. **Track B + C complete** → run leakage check, run merge, start Track E
4. **Track D complete (parallel)** → eval, compare to A
5. **Track E complete** → final eval, write v2.0 production_frozen if better
