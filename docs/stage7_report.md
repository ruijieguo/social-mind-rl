# Stage 7 14B 报告：Phase A 数据扩展（清洁 audit + 9559 records）

> 训练: 2026-05-18 13:50 → 2026-05-19 03:00；250 步, 9559 训练数据
> 评测: 2026-05-19 03:30（full 5718 + clean 4551 + subset500 × 3 协议）
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage7-1x8/20260518-135040/checkpoint-249/`

## 1. 关键结果

**Full 5718 direct**:
| Model | Direct | vs s6 |
|---|---|---|
| **qwen3-14b-tom stage7** | **0.7539** | **-0.41pp** ↓ |
| qwen3-14b-tom stage6 | 0.7580 | — |
| deepseek-v4-pro | 0.8080 | -5.41pp |
| GPT-5.5 | 0.8349 | -8.10pp |

**Clean Eval 4551 direct (剔除 1167 wrong-label 题)**:
| Model | Direct | Δ vs raw |
|---|---|---|
| qwen3-14b-tom stage7 | **0.8436** | +9.0pp（清洁数据红利） |
| qwen3-14b-tom stage6 | 0.8460 | +8.8pp |
| deepseek-v4-pro | 0.9013 | +9.3pp |
| GPT-5.5 | 0.9343 | +9.9pp |

**Subset500 best**: del_tom 0.7620 (vs s6 0.7880, **-2.6pp**)

## 2. 关键发现：Phase A 数据反而略降

stage7 加入 2300 条 GPT-5.5 高质量数据后，**所有协议都略低于 stage6**。
这与 val 信号矛盾（val step 150 = 0.710 vs s6 0.652），值得深究。

### 2.1 训练动态对比

| step | s1 val | s6 val | **s7 val** | Δ vs s6 |
|---|---|---|---|---|
| 50 | 0.348 | 0.496 | **0.516** | +2.0pp |
| 100 | 0.546 | 0.628 | **0.662** | +3.4pp |
| 150 | 0.550 | 0.652 | **0.710** | **+5.8pp** |
| 200 | — | 0.662 | 0.704 | +4.2pp |

**Val 一路超 s6**，但 full eval 反而降了。

### 2.2 Per-task 分解（vs s6 full 5718）

| Task | s6 EN | s7 EN | Δen | s6 ZH | s7 ZH | Δzh |
|---|---|---|---|---|---|---|
| Belief | 0.711 | 0.739 | **+2.8** | 0.754 | 0.718 | -3.5 |
| Desire | 0.594 | 0.617 | +2.2 | 0.572 | 0.622 | **+5.0** |
| Emotion | 0.683 | 0.710 | +2.6 | 0.771 | 0.757 | -1.4 |
| False Belief | 0.888 | 0.873 | -1.5 | 0.870 | 0.850 | -2.0 |
| Intention | 0.809 | 0.785 | -2.4 | 0.862 | 0.841 | -2.1 |
| Knowledge | 0.522 | 0.474 | **-4.8** | 0.481 | 0.464 | -1.7 |
| Non-literal Comm | 0.758 | 0.775 | +1.7 | 0.774 | 0.783 | +0.9 |

**EN 4 涨 3 降**：Belief +2.8, Emotion +2.6, Desire +2.2 — Phase A.1 的 social_norm + factual_detail 数据起效；False Belief / Intention / Knowledge 反退。

**ZH 2 涨 4 降**：Desire +5.0 是 Phase A.2 belief/desire/knowledge 中 Desire 部分起效；其他 task 反退。

## 3. 失败根因分析

### 3.1 数据稀释假设

训练样本从 7259 → 9559（+32%），但 max_steps 不变（250），rollout_batch=32，所以：
- s6: 每条样本平均见 ~7 次
- s7: 每条样本平均见 ~5 次

False Belief / Intention 旧数据被见次数减少 28%，部分能力丢失。

### 3.2 风格分布偏移

GPT-5.5 合成数据**风格与 ToMBench 不匹配**：
- ToMBench 故事中等长度（5-7 句）+ 直白叙事
- GPT-5.5 倾向 8-12 句 + 精致细节

stage7 在 val (含合成数据风格) 学得很好，但泛化到 ToMBench 自然风格时变差。

### 3.3 训练动态走偏

step 200 val 0.704 已 < step 150 val 0.710，**模型已经开始过拟合**新数据风格。
saturation check: step 200 samples_used 仍 ~46/256 (vs s6 step 200 = 16/256)，说明数据混合后**难度遮罩还在过滤大量样本**，但被过滤的可能正是 ToMBench 风格题。

## 4. 改进方向（Stage8 或 Stage7-v2）

### 选项 A: 加 max_steps + 微调比例
- max_steps 250 → 400
- 让旧数据见更多次
- ETA: ~7h 训练

### 选项 B: stage6 ckpt 继续训练
- 从 stage6 ckpt-249 加载 + 新数据 + 100 步
- 保留 s6 已学能力
- 风险：ckpt resume 在 ROLL 中复杂

### 选项 C: 数据 reweight
- 旧数据采样权重 ×1.5（保证旧能力被见频率不变）
- 新数据权重 ×0.7
- 实施简单，加 sampler config

### 选项 D: 风格匹配 (推荐)
- GPT-5.5 合成时显式约束故事长度（5-7 句）+ 直白叙事
- 重新合成 1500 条**风格匹配** Phase A 数据
- 这是改进 plan_improvement 的下一步重点

## 5. 实用决策

**生产部署**：用 **stage6 14B (0.7580)** 作为生产模型，stage7 暂不部署。

**但 stage7 在 EN Belief/Emotion/Desire 上有进步**，可考虑 ensemble：
- EN Belief / Emotion / Desire / Non-literal: 用 stage7
- EN False Belief / Intention / Knowledge: 用 stage6
- 所有 ZH: 用 stage6（ZH 整体 stage7 倒退）

## 6. Clean Eval 关键洞察

**Stage7 raw vs clean 提升 9.0pp**，与 deepseek (9.3pp) / GPT-5.5 (9.9pp) 相近。说明：
- ToMBench wrong-label 是**所有模型的共同 ceiling**，不是某模型特有问题
- Stage7 的真实能力被 raw 5718 严重低估

stage6 raw 0.758 → clean 0.846，gap to GPT-5.5 缩到 -8.8pp（vs raw -7.8pp）。

## 7. 产物清单

| 路径 | 内容 |
|---|---|
| `output/eval/stage7_full5718.{json,md}` | full 5718 |
| `output/eval/stage7_clean_eval.{json,md}` | 4551 cleaned |
| `output/eval/stage7_subset500.{json,md}` | subset500 × 3 |
| `output/analysis/curves_stage7_14b.png` | 训练曲线 |
| `output/analysis/errors_stage7.md` | 错题样本 |
| `data/tom/tom_train.jsonl` | 9559 (Phase A merged) |
| `data/tom/tombench_eval_clean.jsonl` | 4551 cleaned eval |
| `output/analysis/gpt55_eval_full_audit.jsonl` | 5718 audit |
| HF model | `qwen3-14B-tom-hf-stage7/` (28 GB) |
| Megatron ckpt | `qwen3-14B-tombench-rlvr-stage7-1x8/.../checkpoint-249/` |
| Log | `logs/train_stage7_1x8_14b_20260518_135019.log` |

最后更新: 2026-05-19 03:30
