# Stage5 1×8 训练 + 评测报告

> 训练: 2026-05-16 20:00 → 2026-05-17 08:25 (UTC); 250 步, **8901 records** (含 Phase-1 fixed 数据)
> Eval: 2026-05-17 09:00 (full 5718 + subset500 × 3 protocols)
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage5-1x8/20260516-200108/checkpoint-249/`

## 1. Headline 数字

**Full ToMBench 5718 (direct)**:

| Model | Overall | EN | ZH |
|---|---|---|---|
| qwen3-8b-nt baseline | 0.7009 | 0.7020 | 0.6999 |
| qwen3-8b-tom **stage1** | **0.7394** | 0.7275 | **0.7513** |
| qwen3-8b-tom stage2 | 0.7263 | 0.7223 | 0.7303 |
| qwen3-8b-tom stage3 | 0.7302 | 0.7156 | 0.7447 |
| qwen3-8b-tom **stage5** | 0.7305 | 0.7181 | 0.7429 |
| deepseek-v4-pro target | 0.7880 | 0.7803 | 0.7966 |

**Stage5 0.7305 与 stage3 0.7302 几乎平等**，比 stage1 0.7394 还低 0.89pp。

**Subset500 (3 protocols)** —— Stage5 的强项：

| Protocol | stage1 | stage2 | stage3 | **stage5** |
|---|---|---|---|---|
| direct | **0.7460** | 0.7340 | 0.7360 | 0.7420 |
| cot | 0.6980 | **0.7540** | 0.6240 | 0.7380 |
| del_tom | 0.7460 | 0.7480 | 0.7080 | **0.7520** ← 最高 |
| **Best** | direct 0.7460 | cot 0.7540 | direct 0.7360 | **del_tom 0.7520** |

**Stage5 三个协议都表现接近** (0.738-0.752)，最佳 del_tom 0.7520。但还是不如 stage2 cot 0.7540。

## 2. Per-task 对比 (Full 5718, direct)

| Task | baseline | stage1 | stage2 | stage3 | **stage5** | s5 vs ds gap |
|---|---|---|---|---|---|---|
| Belief | 0.6725 | 0.6937 | 0.6373 | **0.7007** | 0.6972 | -10.28pp |
| Desire | 0.5861 | 0.5917 | 0.5861 | 0.5556 | 0.5611 | -7.78pp |
| Emotion | 0.6893 | **0.7286** | 0.7012 | 0.7107 | 0.7107 | +0.14pp ✓ |
| False Belief | 0.7277 | 0.8520 | 0.8385 | **0.8649** | 0.8412 | -2.03pp |
| Intention | 0.7500 | 0.7647 | 0.7632 | 0.7618 | **0.7691** | -4.45pp |
| Knowledge | 0.4810 | 0.4792 | **0.4879** | 0.4377 | 0.4792 | -12.08pp |
| Non-literal Comm | 0.7767 | **0.7674** | 0.7553 | 0.7540 | 0.7587 | -8.46pp |

**关键洞察**:
- Stage5 在所有 task 上**没有取得 Phase-1 数据应该带来的提升**
- Knowledge 0.4792 (与 stage1 一致) — Phase-1 scalar 数据没产生迁移
- Non-literal Comm 0.7587 (vs stage1 0.7674) — Phase-1 faux_pas + hinting 数据没修复
- **唯一进步**: Intention 0.7691 (vs stage1 0.7647, +0.44pp) — hinting 数据可能起到一点作用

## 3. 训练动态对比

**Val 轨迹 (subset500)**:
| step | stage1 val_all | stage2 | stage3 | stage4 | **stage5** |
|---|---|---|---|---|---|
| 0 | 0.042 | 0.036 | 0.038 | 0.034 | 0.040 |
| 50 | 0.204 | 0.206 | 0.148 | 0.142 | 0.184 |
| 100 | 0.454 | 0.466 | 0.222 | 0.154 | 0.428 |
| 150 | 0.548 | 0.546 | 0.236 | — | **0.582** ← 最高 |
| 200 | — | 0.530 | 0.226 | — | **0.600** ← 最高 |

**Stage5 val 在 step 150-200 达到所有 stage 中的最高**，但 final eval (full 5718) 反而平庸。这反映 **subset500 上的强势没有完全迁移到 5718 的 70+ subtasks**。

## 4. 结论：Phase-1 合成数据收益不显著

**核心结论**: 1100 条 Phase-1 合成数据 (faux_pas/scalar/hinting/so_belief 4 类) 没显著推动 stage5 超越 stage1。

**为什么**:
1. **训练集 vs 测试集分布差距**: Phase-1 数据由 deepseek-v4-flash/pro 合成，模式与 ToMBench 仍有差异
2. **数据量太少**: 1100 条占总 8902 的 12%，但 subset500 上看到 val 突破说明训练 prompt 学会了，只是没迁移
3. **过拟合 Phase-1 模式**: stage5 在 subset500 (含 ToMBench 题型) 上表现好，但 5718 全集中 task variants 多，泛化失败
4. **deepseek 合成的 faux_pas 题模式可能太单一** (我们 fixed 了空 C/D 但内容多样性不足)

## 5. 全部 5 个 stage 总结

| Stage | Overall (5718, direct) | Best Protocol | 训练时长 | 备注 |
|---|---|---|---|---|
| baseline | 0.7009 | — | — | qwen3-8b-nt |
| **stage1** | **0.7394** | direct 0.7460 | 3h 20m | 200 steps, 4k data |
| stage2 | 0.7263 | cot 0.7540 | 8h 24m | 500 steps, 8k data, overfit |
| stage3 | 0.7302 | direct 0.7360 | 5h 24m | KL=true, response=384 |
| stage4 | (failed) | — | 2h killed | KL+empty C/D 数据破坏 |
| stage5 | 0.7305 | del_tom 0.7520 | 4h 25m | KL=false, Phase-1 fixed |
| deepseek | 0.7880 | direct 0.7880 | — | 目标 |

**Best single number 至今**: **stage1 direct 0.7394** 或 **stage2 cot 0.7540**

## 6. 部署

- **HF 模型路径**: `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage5/` (16 GB)
- **vLLM serve**: `qwen3-tom-serve-stage5` (port 8000), model id `qwen3-8b-tom-stage5`

## 7. 下一步建议

经过 5 个 stage 实验，**靠数据合成 + RL 训练逼近 deepseek 的 0.7880 已经触及瓶颈**。剩余 4-5pp gap 来自:

1. **模型 capacity**: Qwen3-8B vs deepseek 几百 B 参数差距，单纯 RL 难以弥补
2. **deepseek 自身合成数据局限**: 用 deepseek 合成的 ToM 题学不到 deepseek 自己也不擅长的 task (Knowledge 31% both-wrong)
3. **ToMBench label 偏差**: 13% questions deepseek 也答错 = hard ceiling

**实用方案**:
- **生产部署用 stage1** (direct best 0.7394) 或 **stage2 cot** (0.7540) 配合 protocol routing
- 距 deepseek 4-5pp 的 gap 接受为合理的"小模型 RL 上限"
- 想突破 0.78+ 需要：(a) 换更大 base model, (b) 更高质量数据 (人工标注 ToM 题 ≥ 5000 条)

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `output/eval/stage5_full5718.{json,md}` | 5718 direct |
| `output/eval/stage5_subset500.{json,md}` | 500 × 3 protocols |
| `output/analysis/curves_stage5_1x8.png` | 训练曲线 |
| `output/analysis/errors_stage5.md` | 错题样本 |
| `logs/train_stage5_1x8_20260516_200048.log` | 完整训练日志 (13 MB) |
| HF model: `qwen3-8B-tom-hf-stage5/` | 部署用 (16 GB) |

Last updated: 2026-05-17 09:00
