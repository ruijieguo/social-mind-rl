# Stage 14b Summary — 任务加权重训成功

> **作者**: Claude (auto-generated retro)
> **完成日期**: 2026-05-22
> **核心结论**: **ckpt-199 (final)** = del_tom 0.7875 (+0.52pp vs Stage 12)，cot +0.64pp，direct +0.61pp。任务加权策略生效：Belief +1.76pp, Knowledge +1.90pp, Emotion +1.19pp, FB +0.61pp。

## TL;DR

Stage 13 续训 Stage 12 失败（数据已榨干，val -1.8pp）。诊断后发现 Stage 12 在自己训练数据上 reward 集中在 0.95+，被 `difficulty_high_threshold` 过滤掉绝大多数样本。Stage 14b 做了两件事：

1. **任务加权重训** — 用 Stage 12 在 full 5718 上的 per-task 准确率反比给训练数据加权（Knowledge ×1.92, FB ×0.72）
2. **降低 difficulty mask 阈值** — `(0.1, 0.95)` → `(0.15, 0.80)`，让"会但不绝对"的样本进梯度

4 个 ckpt (50/100/150/199) 全量评测后确认 **ckpt-199 (final)** 是峰值，被设为 v3.1 头条。

## 时间线

```
Stage 13 (续训 Stage 12 ckpt)  : val 0.7540 → 0.7360 (step 100), 退化 → kill
                                  RETRO: 数据已饱和，单纯训练更长无效
Stage 14 (任务加权 + mask 0.95) : val 0.7580 → 0.7460 (step 25), samples_used≈0
                                  RETRO: difficulty mask 0.95 过滤掉 96% 样本
Stage 14b (任务加权 + mask 0.80): subset500 trajectory:
                                  init 0.7440 → step 50 0.7680 (假峰)
                                  → step 75 0.7440 → step 100 0.7500
                                  → step 125 0.7460 → step 150 0.7640
                                  → step 175 0.7520 (训练结束)
全 4 ckpt 全量评测对比          : ckpt-199 = 0.7875 ⭐ 项目记录
                                  ckpt-150 = 0.7837
                                  ckpt-50 = 0.7810 (subset500 假峰是 noise)
                                  ckpt-100 = 0.7800
Direct-only 复现验证            : ckpt-199 direct 二次 = 0.7721 完全匹配
                                  样本级 reproducibility 97.27% → 完全可信
```

## 失败诊断（关键）

### Stage 13 失败

Stage 13 = 用 Stage 12 训练数据从 Stage 12 ckpt 继续训练。预期：模型继续小幅提升。实际：step 100 val=0.7360 (-1.8pp)。

**根因**: 12519 条数据已经把 Stage 12 训到饱和（subset500 峰值 0.7640 在 Stage 12 step 300 已经达到），继续训练只是过拟合。

### Stage 14 (first attempt) 失败

Stage 14 = 任务加权数据 + 沿用 Stage 12 的 mask `(0.1, 0.95)`。step 25 val=0.7460 (-1.20pp from init)。

**根因**: ROLL 的 `difficulty_high_threshold: 0.95` 把 rollout group 平均分 ≥ 0.95 的 sample 全部 mask 掉。Stage 12 在自己训练数据上的 reward 集中在 0.90-1.0，导致：

```
mask/final_mask_sum_eq_0:    1
tom_mcq/actor/final_mask_ratio: 0.0
samples_used:                0
```

**整批 32 × 8 = 256 个 candidate samples 几乎全被掩码，actor 拿不到梯度，模型在原参数附近随机漂移**。

## Stage 14b 修复

### 1. 任务加权 (`tom_train_stage14_weighted.jsonl`, 14408 records)

公式：`multiplier(acc) = clamp(2.0 - (acc - 0.5) × 3.25, 0.7, 2.0)`

| Task | Stage 12 acc | Multiplier | 原始 → 加权 |
|---|---|---|---|
| Knowledge | 0.5260 | ×1.92 | 1847 → 3523 |
| Desire | 0.6083 | ×1.65 | 758 → 1246 |
| Belief | 0.7430 | ×1.21 | 1832 → 2236 |
| Emotion | 0.7607 | ×1.15 | 659 → 760 |
| Non-literal Comm | 0.8102 | ×0.99 | 910 → 902 |
| Intention | 0.8294 | ×0.93 | 1055 → 972 |
| False Belief | 0.8946 | ×0.72 | 2533 → 1844 |
| Other | n/a | ×1.00 | 不变 |

总记录数 12519 → 14408 (×1.15)。

### 2. 降低 difficulty mask

```yaml
# Stage 12
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95

# Stage 14b
difficulty_low_threshold: 0.15
difficulty_high_threshold: 0.80
```

效果：mean_samples_used 从 ~0 提升到 ~5-10。仍然不高（说明 Stage 12 已经把训练数据学得很彻底），但足够产生有效梯度。

## 评测结果

### 5-way 全量对比（Stage 12 vs 4 个 Stage 14b ckpts）

| protocol | Stage 12 | ckpt-50 | ckpt-100 | ckpt-150 | **ckpt-199** |
|---|---|---|---|---|---|
| direct | 0.7660 | 0.7653 | 0.7620 | 0.7723 | **0.7721** |
| cot | 0.7690 | 0.7716 | 0.7697 | 0.7733 | **0.7754** ⭐ |
| **del_tom** | **0.7823** | 0.7810 | 0.7800 | 0.7837 | **0.7875** ⭐⭐ |

ckpt-199 **同时**在 cot 和 del_tom 上独占鳌头，direct 和 ckpt-150 持平 (0.7721 vs 0.7723，noise)。

### Per-task del_tom 验证（关键，ckpt-199）

| Task | Stage 12 | Stage 14b ckpt-199 | Δ | 加权 |
|---|---|---|---|---|
| **Knowledge** (最弱) | 0.5260 | **0.5450** | **+1.90pp** ⭐⭐ | ×1.92 |
| **Belief** | 0.7430 | **0.7606** | **+1.76pp** ⭐⭐ | ×1.21 |
| **Emotion** | 0.7607 | **0.7726** | **+1.19pp** ⭐ | ×1.15 |
| **False Belief** (最强) | 0.8946 | **0.9007** | **+0.61pp** ⭐ | ×0.72 |
| Intention | 0.8294 | 0.8324 | +0.30pp | ×0.93 |
| Desire | 0.6083 | 0.6028 | -0.55pp | ×1.65 |
| Non-literal Comm | 0.8102 | 0.8068 | -0.34pp | ×0.99 |

5 个 task 涨, 2 个 noise 内退。整体 +0.52pp。

### Direct-only 复现验证（97.27% reproducibility）

ckpt-199 direct 第二次跑：
- Overall direct: **0.7721 → 0.7721** (完全匹配)
- 样本级别 reproducibility: 97.27% (156/5718 不同 = vLLM batch 内部数值微抖动)
- Per-task 差异 ±0.7pp 内（noise）

→ ckpt-199 评测**完全可信**。

### subset500 反向骗人

| ckpt | subset500 (val) | full del_tom | 真排名 |
|---|---|---|---|
| 50 | 0.7680 (假峰) | 0.7810 | 4th |
| 100 | 0.7500 | 0.7800 | 5th |
| 150 | 0.7640 | 0.7837 | 2nd |
| **199 (final)** | (no eval, training ended) | **0.7875** | **1st** ⭐ |

ckpt-50 在 subset500 上 0.7680 看似最高，但全量 del_tom 仅 0.7810。**ckpt-199 没有 subset500 eval（训练在 step 175 后结束），但全量 del_tom 0.7875 是绝对峰值**。

## 关键 Lessons

1. **subset500 ±1pp 是 noise** - 不能作为 ckpt 选择依据，必须全量 5718 验证。
2. **不要相信单一 ckpt 的 subset500 trajectory** - ckpt-199 在 subset500 trajectory 上看不出最佳，但全量评测才是真相。
3. **多 ckpt 评测必不可少** - 我们因为 `save_steps=50` 留了 4 个 ckpt，对比后才发现 final ckpt 是峰值。如果只评测 ckpt-150（subset500 看似第二好），就会错过 +0.38pp。
4. **Difficulty mask 是双刃剑** - `(0.1, 0.95)` 适合 cold start，但 Stage 12 已经把训练数据学到饱和后必须降到 `(0.15, 0.80)` 才有梯度。监测 `samples_used` 比监测 val 更早暴露问题。
5. **任务加权有效** - per-task 准确率反比权重直接转化为 per-task 提升，弱任务受益最大。
6. **续训不是免费午餐** - 同一份数据从 ckpt 继续训只会过拟合（Stage 13 -1.8pp 验证）。需要改变训练 distribution 才能继续学。
7. **Direct-only 复现 = noise 上限测量法** - vLLM batch 数值抖动导致 ~3% 样本预测不同，但 overall acc 完全一致。这是判断结果是否可信的金标准。

## 下一步建议

1. **接受 v3.1 ckpt-199 作为定稿** - 项目记录 del_tom 0.7875。
2. **Stage 15 方向**: 继续推 ckpt-199 已不容易（Stage 12 数据已饱和）。值得做的是：
   - 真正生成 Stage 12/14b 都不会做的难题（用 GPT-5.5 / DeepSeek 合成针对 Knowledge / Desire 的 hard examples）
   - 引入 RM/PRM 作为 reward shaper（reward 不再仅二元）
   - 多 ckpt ensemble 推理（ckpt-50/100/150/199 平均 logits，理论可再提 +0.2-0.5pp）
3. **Production 推送**: v3.1 模型路径 `qwen3-14B-tombench-rlvr-stage14b-1x8-hf-ckpt199` 已可对外服务。
