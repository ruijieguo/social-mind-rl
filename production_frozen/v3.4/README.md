# v3.4 Production Frozen — Stage 18 ckpt-30 (GPT-5.5 Distillation)

> **Frozen on**: 2026-05-26
> **Predecessor**: v3.3 (Stage 17 ckpt-120)
> **Method**: GPT-5.5 paraphrase + verify distillation (925 records → 734 retained after dedup)

## Eval Results (4 benchmark × 3 protocol)

| Bench | direct | cot | del_tom |
|---|---|---|---|
| ToMBench (5718) | 0.7674 | 0.7674 | 0.7777 |
| Hi-ToM (600) | 0.5833 | **0.7433** ⭐ | **0.7617** ⭐ |
| SocialIQA (1954) | 0.7866 | 0.7774 | 0.7861 |
| EmoBench (1200) | 0.6508 | 0.6450 | 0.6633 |

**Hi-ToM cot 距 DeepSeek 仅 -0.42pp** (v3.1 -8.08pp → v3.4 -0.42pp 是质变)

## Distillation method

1. v3.3 错例提取: EmoBench 204 + SocialIQA 237 + ToMBench Belief+Knowledge 359 + Hi-ToM 125 = 925
2. GPT-5.5 paraphrase Q + GPT-5.5 verify gold (T=0)
3. Keep only samples where GPT-5.5 verified gold (kept rate ~50%) → 734 records
4. MinHash dedup → 19033 final stage18 train set
5. RL continue from v3.3, 150 steps, ckpt-30 best @ val_all=0.6875

## Distillation effectiveness

✅ **Hi-ToM (推理 task)**: cot +0.83pp, del_tom +1.17pp — distillation 真有效
⚠️ **EmoBench / SocialIQA (knowledge task)**: 几乎持平 — base 14B 缺少 6-opt 情绪命名能力，蒸馏教不会
❌ **ToMBench**: -0.19~-0.24pp — 蒸馏数据稀释 backbone（与 v3.2/v3.3 同样模式）

## Files

| Path | Purpose |
|---|---|
| `configs/rlvr_config_14b_stage18_FROZEN.yaml` | 训练配置 |
| `scripts/distill_gpt55.py` | GPT-5.5 蒸馏 pipeline |
| `scripts/run_tombench.py` `scripts/run_generic_mcq.py` `scripts/extractors_generic.py` | Eval framework |
| `scripts/tom_mcq_reward_worker_FROZEN.py` | reward worker |
| `eval/stage18_ckpt30_*.json` | 4 benchmark raw eval results (full size) |
| `docs/full_eval_report_2026-05-24.md` | 9-model 完整对比报告 |

## Verify integrity

```bash
./verify.sh
```

## Reproduce

ckpt-30 HF model: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage18-1x8-hf-ckpt30`
