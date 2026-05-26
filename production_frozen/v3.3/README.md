# v3.3 Production Frozen — Stage 17 ckpt-120

> **Frozen on**: 2026-05-26
> **Predecessor**: v3.2 (Stage 16 ckpt-270)
> **Successor**: v3.4 (Stage 18 ckpt-30)

## Eval Results (4 benchmark × 3 protocol)

| Bench | direct | cot | del_tom |
|---|---|---|---|
| ToMBench (5718) | 0.7658 | 0.7698 | 0.7796 |
| Hi-ToM (600) | 0.5817 | 0.7350 | 0.7500 |
| SocialIQA (1954) | 0.7866 | 0.7825 | 0.7830 |
| EmoBench (1200) | 0.6483 | 0.6483 | 0.6700 |

vs DeepSeek-v4-pro: -1.25pp on Hi-ToM cot (closest), -11pp on EmoBench

## Training recipe (key delta vs v3.2)

- Base: Stage 16 ckpt-270
- Data: stage17 (Hi-ToM direct-style + EmoBench/SocialIQA synth, 19033 records)
- 150 steps, lr=1e-6, GRPO + DAPO Clip-Higher

## Files

| Path | Purpose |
|---|---|
| `configs/rlvr_config_14b_stage17_FROZEN.yaml` | 训练配置 |
| `scripts/build_hitom_train_direct.py` | Hi-ToM direct-style 数据 |
| `scripts/synth_emobench_socialiqa.py` | EmoBench/SocialIQA 合成 |
| `scripts/run_tombench.py` `scripts/run_generic_mcq.py` `scripts/extractors_generic.py` | Eval framework |
| `scripts/tom_mcq_reward_worker_FROZEN.py` | reward worker (A-Z support, l_max_long) |
| `eval/stage17_ckpt120_*.json` | 4 benchmark raw eval results |
| `docs/full_eval_report_2026-05-24.md` | 完整对比报告 |

## Verify integrity

```bash
./verify.sh
```

## Reproduce

ckpt-120 HF model 在 TRAIN host: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage17-1x8-hf-ckpt120`
