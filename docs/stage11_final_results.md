# Stage 11 Final Results (TBD — fills in as tracks complete)

> Headline TBD: 14B 0.7594 → ? (target 反超 deepseek 0.8080)

## TL;DR

The Stage 11 plan executed in 5 parallel tracks:
- **Track A** (eval-time, no training): del_tom voting → **+2.16pp** free win, 0.7594 → 0.7810.
- **Track B** (program-guided data): 2000 ExploreToM v2 records with belief annotations.
- **Track C** (HOT-targeted GPT-5.5): ~1260 records targeting per-task gaps.
- **Track D** (continue training stage 8): control experiment for "more steps alone".
- **Track E** (stage 12 integration): stage 8 + B + C training.

Final result: **TBD** (depends on Track D + E completion).

## 1. Track A: eval-time protocol gain ✅

Stage 8 14B unchanged model, full 5718 eval, 3 protocols:

| Protocol | Acc | Δ vs direct |
|---|---|---|
| direct | 0.7590 | — |
| cot | 0.7681 | +0.91pp |
| **del_tom (N=8 voted CoT)** | **0.7810** | **+2.20pp** |

Per-task del_tom gains are concentrated in structural ToM:

| Task | direct | del_tom | gain |
|---|---|---|---|
| **False Belief** | 0.8622 | **0.9115** | **+4.93pp** |
| Belief | 0.7394 | 0.7676 | +2.82pp |
| Desire | 0.5694 | 0.5972 | +2.78pp |
| Emotion | 0.7274 | 0.7488 | +2.14pp |
| Intention | 0.8176 | 0.8324 | +1.47pp |
| Non-literal Comm | 0.7921 | 0.7961 | +0.40pp |
| Knowledge | 0.5138 | 0.5156 | +0.17pp |

**Lesson**: voting helps *belief tracking* tasks but doesn't help *factual recall* (Knowledge) or *pragmatic understanding* (Non-literal). The variance reduction targets the right structure.

## 2. Per-task gaps vs SOTA (from Track A direct)

Stage 8 del_tom (0.7810) compared to deepseek-v4-pro (0.8080) and GPT-5.5 (0.8349):

| Task | s8_del | deepseek | gpt-5.5 | gap to deepseek |
|---|---|---|---|---|
| **False Belief** | **0.9115** | 0.8946 | 0.9264 | **+1.69pp ⭐** |
| Non-literal Comm | 0.7961 | 0.8128 | 0.8342 | -1.67pp |
| Desire | 0.5972 | 0.6333 | 0.6806 | -3.61pp |
| Knowledge | 0.5156 | 0.5675 | 0.6713 | -5.19pp |
| Emotion | 0.7488 | 0.8048 | 0.8155 | -5.60pp |
| Intention | 0.8324 | 0.8926 | 0.8794 | -6.03pp |
| **Belief** | 0.7676 | 0.8486 | 0.8415 | **-8.10pp ⚠** |

This map drove the Stage 12 data targeting (Track B + C).

## 3. Track B: ExploreToM v2 ✅

**Pivot**: The original plan was to run Meta's ExploreToM A* search using stage 8 14B as the teacher. Stage 8 (MCQ-tuned) cannot generate the multi-step story contexts the framework requires. We pivoted to using **Meta's official `ExploreToM-data-sample.csv`** (13,309 records).

Filter: kept only records where Llama-3.1-70B accuracy ≤ 0.7 (hard examples). Converted to ToMBench MCQ format with smart distractor extraction.

**Output**: `data/tom/raw/exploretom_v2.jsonl` — 2000 records.

| Task | Records |
|---|---|
| Belief | 1152 |
| False Belief | 724 |
| Knowledge | 124 |

Each record retains belief metadata (`nth_order` 1/2, `is_fb_1st/2nd`, `llama70b_acc_infilled`, `story_type`) for difficulty-aware curriculum.

Leakage: 0/2000 (jaccard < 0.85 vs eval set, hash-based dedup).

## 4. Track C: HOT-targeted GPT-5.5 synth 🟡 (TBD)

Analysis of 492 HOT records (where stage 8 fails but both deepseek + GPT-5.5 succeed) revealed per-task error patterns:

| Task | HOT count |
|---|---|
| False Belief | 102 |
| Non-literal Comm | 92 |
| Knowledge | 84 |
| Emotion | 80 |
| Intention | 61 |
| Desire | 39 |
| Belief | 34 |

GPT-5.5 generates ~180 same-pattern records per task in different scenarios.

**Output**: `data/tom/raw/synth_gpt55_phase_d_hot.jsonl` — TBD records (target ~1260).

Cost: ~$50.

## 5. Track D: continue stage 8 training 🟡 (TBD)

Identical config to stage 8, only `pretrain` changed to stage 8 HF. Tests whether stage 8 was already at plateau.

**Trajectory** (so far):
| step | val |
|---|---|
| 0 (init) | 0.7080 (= stage 8 step 200) |
| 50 | 0.7200 (+1.20pp) |
| 100 | TBD |
| 150 | TBD |
| 200 | TBD |
| 350 | TBD |

Final result: TBD.

## 6. Track E: Stage 12 integration ⏳ (queued)

Stage 12 = stage 8 + Track B + Track C, identical recipe to stage 8 (350 GRPO steps).

Data composition (as of Track C completing):
```
Total: ~12,500 records (0 leakage vs eval)
  Base (stage 8): 9,259
  ExploreToM v2:  2,000
  HOT synth:      ~1,260
```

Per-task data emphasis (after merge):
| Task | Base | New | Total |
|---|---|---|---|
| Belief | 680 | 1152 + ~180 | 2012 (focus) |
| False Belief | 1629 | 724 + ~180 | 2533 |
| Knowledge | 1543 | 124 + ~180 | 1847 |
| Intention | 814 | 0 + ~180 | 994 (more EN/ZH) |
| Emotion | 530 | 0 + ~180 | 710 |
| Desire | 719 | 0 + ~90 | 809 |
| Non-literal Comm | 638 | 0 + ~180 | 818 |

Final result: TBD.

## 7. Resource summary

| Track | GPU | API | Wall time |
|---|---|---|---|
| A | 1 GPU | 0 | 30 min |
| B | 0 (data DL) | 0 | 5 min |
| C | 0 (DEV) | $50 | ~5h |
| D | 8 GPU | 0 | ~5h |
| E | 8 GPU | 0 | ~5h |

**Total**: ~80 GPU-h, $50 API, ~16h wall time (with parallelism).

## 8. Headline expected

If Track D adds +1pp and Stage 12 adds another +2pp, with del_tom layered:
- Stage 12 raw: 0.78-0.80 (vs s8 0.7594)
- Stage 12 + del_tom: **0.80-0.83** (target reached)

**Stretch target**: 0.83+ would tie or exceed GPT-5.5 (0.8349) on full 5718.
