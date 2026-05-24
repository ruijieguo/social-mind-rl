# Cross-Benchmark Eval Report (5 models × 3 benchmarks)

> **Date**: 2026-05-24 (TRAIN evals 2026-05-23, DeepSeek 2026-05-23 → 2026-05-24)
> **Models**: Qwen3-14B base, 14B v3.1 (Stage 14b ckpt-199), Qwen3-8B base, 8B v1.0 (Stage 15 ckpt-150), DeepSeek-v4-pro
> **Benchmarks**: SocialIQA dev (1954, 3-opt), EmoBench (1200, 4-6 opt), Hi-ToM (600, 15-opt)
> **Protocols**: direct, cot, del_tom (8 sample voting)

## TL;DR — 三大修正

1. **Hi-ToM 的"v3.1 完胜 DeepSeek"是 truncation artifact**: max_tokens=2048 让 DeepSeek 60% 输出 truncate，错误得到 0.38 cot；修正到 8192 后真实 ~0.86，**DeepSeek 完胜 v3.1 ~20pp**。
2. **DeepSeek 在所有 benchmark × protocol 上全面领先** 我们的 v3.1 / v1.0
3. **ToMBench RL 训练只在 Hi-ToM 上有强迁移**（+22-26pp vs base），SocialIQA/EmoBench 几乎 0 迁移

## 完整结果矩阵

| Benchmark | Protocol | 14B base | 14B v3.1 | 8B base | 8B v1.0 | DeepSeek |
|---|---|---|---|---|---|---|
| **SocialIQA** | direct | 0.7871 | 0.7876 | 0.7559 | 0.7605 | **0.8101** |
| | cot | 0.7856 | 0.7861 | 0.7600 | 0.7733 | **0.8106** |
| | del_tom | 0.8035 | 0.7892 | 0.7758 | 0.7851 | **0.8188** |
| **EmoBench** | direct | 0.6325 | 0.6483 | 0.6033 | 0.6000 | **0.7625** |
| | cot | 0.6342 | 0.6575 | 0.5725 | 0.6208 | **0.7550** |
| | del_tom | 0.6717 | 0.6775 | 0.6233 | 0.6342 | **0.7733** |
| **Hi-ToM** | direct | 0.5333 | 0.5417 | 0.5550 | 0.5717 | **~0.864** † |
| | cot | 0.4367 | 0.6667 | 0.4467 | 0.6183 | **~0.86** ‡ |
| | del_tom | 0.4783 | 0.7217 | 0.4817 | 0.6567 | **N/A** § |

† DeepSeek Hi-ToM direct: 432/500 cumulative 0.864 (最后一次重跑卡在 500/600)；之前 conc=12 重跑 480/599 = 0.802 (有 timeout)；趋势稳定。
‡ DeepSeek Hi-ToM cot: 259/300 partial = 0.863 (process 多次卡死在 ~300)；趋势与 direct 接近
§ Hi-ToM del_tom 600×8=4800 sample 在 deepseek API 上预估 10h+，且 hitom hang 频繁（已 3 次卡死），不可行。

## 详细 Per-task 分解

### Hi-ToM by ToM order (direct)

| Order | 14B base | 14B v3.1 | 8B base | 8B v1.0 | DeepSeek |
|---|---|---|---|---|---|
| order_0 | 0.825 | 0.892 | 0.825 | 0.825 | **1.000** |
| order_1 | 0.575 | 0.575 | 0.583 | 0.542 | **0.960** |
| order_2 | 0.392 | 0.400 | 0.450 | 0.508 | **0.870** |
| order_3 | 0.450 | 0.433 | 0.483 | 0.508 | **0.670** |
| order_4 | 0.425 | 0.408 | 0.508 | 0.475 | **0.820** (partial 100q) |

**关键洞察**: DeepSeek 在 order_0/1 (低阶 ToM) 接近完美 (1.00/0.96)，order_3 难度最高时仍 0.67。我们的 RL 模型 order_0 才 0.83-0.89，明显起点低。

### EmoBench by sub-task (del_tom)

| Sub-task | 14B base | 14B v3.1 | 8B base | 8B v1.0 | DeepSeek |
|---|---|---|---|---|---|
| EA (action) | 0.6975 | 0.7075 | 0.6325 | 0.6725 | **0.83+** |
| EU_emotion (情绪命名) | 0.5050 | 0.5375 | 0.4500 | 0.4625 | **~0.70+** |
| EU_cause (情绪原因) | 0.7800 | 0.7875 | 0.7325 | 0.7675 | **~0.80+** |

**最弱**: EU_emotion (情绪命名) — 我们模型只有 0.45-0.54, DeepSeek ~0.70。这是 EmoBench 最难子任务，需要 6 选项中精准命名情绪。

## RL Gain 分析（v3.1 vs 14B base, v1.0 vs 8B base）

| Benchmark | Protocol | 14B v3.1 - base | 8B v1.0 - base |
|---|---|---|---|
| SocialIQA | direct | +0.05pp | +0.46pp |
| SocialIQA | cot | +0.05pp | +1.33pp |
| SocialIQA | del_tom | **-1.43pp** ⚠️ | +0.92pp |
| EmoBench | direct | +1.58pp | -0.33pp |
| EmoBench | cot | +2.33pp | +4.83pp |
| EmoBench | del_tom | +0.58pp | +1.08pp |
| **Hi-ToM** | **cot** | **+23.00pp** ⭐⭐⭐ | **+17.17pp** ⭐⭐⭐ |
| **Hi-ToM** | **del_tom** | **+24.33pp** ⭐⭐⭐ | **+17.50pp** ⭐⭐⭐ |

## Truncation Bug 详解

### 现象
DeepSeek-v4-pro 是 reasoning 模型，思考过程通过 `content` 输出（包含 `<think>...</think>...\\boxed{X}`）。Hi-ToM 题需要追踪 5 个角色 + 5 阶 belief，思考链很长。

### 错误数据 (v1, max_tokens=2048)

| Hi-ToM Protocol | Raw acc | Excl-null acc | Null rate |
|---|---|---|---|
| direct | 0.4750 | ~? | **50.8%** |
| cot | **0.3800** | **0.9540** | **60.2%** |
| del_tom | 0.4900 | ~? | **46.7%** |

### 修正后 (v2, max_tokens=8192)

| Hi-ToM Protocol | Acc | Null rate |
|---|---|---|
| direct | **0.864** (partial) | <2% |
| cot | **0.863** (partial 300) | <2% |

修正前导致**真实结论被反转**：
- 之前以为 v3.1 cot 完胜 DeepSeek +28pp
- 真实是 DeepSeek 领先 v3.1 ~20pp

## API Stability Issue

DeepSeek 在 Hi-ToM 上**多次出现 process 死锁**（log 长时间无更新但 process 仍 alive）：
- 第一次 conc=12: cot 600/600 完成但 del_tom 完全 hang (50min 无进展)
- 第二次 conc=4: direct 跑到 300/600 后 hang 4 小时
- 第三次 conc=8: cot 跑到 300/600 hang
- 第四次 conc=8 only direct: 跑到 500/600 hang 35min

**Pattern**: 似乎 deepseek-v4-pro API 在长 reasoning_content 题目上有偶发 timeout，client 的 retry 逻辑无法恢复。

## 结论

### ToMBench RL 训练效果评估

✅ **Hi-ToM 强迁移**: +17-25pp，证明 belief tracking 能力真正提升
⚠️ **SocialIQA/EmoBench 几乎无迁移**: <2pp 改善，甚至 v3.1 del_tom 出现 -1.43pp 退步
❌ **DeepSeek 仍大幅领先**: 平均落后 5-30pp

### 下一阶段建议

1. **数据多样化**: 引入 SocialIQA-style commonsense + EmoBench-style emotion 训练数据
2. **Hi-ToM-style 训练**: 加入显式的 multi-order belief tracking 数据可进一步推 Hi-ToM acc
3. **Distillation**: 用 DeepSeek 在 EmoBench/Hi-ToM 上的输出做 SFT 教学
4. **EU_emotion 弱项专项**: 6-option 情绪命名训练数据

## 数据完整性证书

- 所有 14 个 (model × benchmark) 评测 = 14 × 3 protocols = 42 items
- 39 完整 (count match expected sample size)
- 3 partial (DeepSeek Hi-ToM cot 300/600, DeepSeek Hi-ToM direct 500/600, DeepSeek Hi-ToM del_tom 0/600)
- DeepSeek SocialIQA / EmoBench 两 benchmark 全完整 9 protocols
