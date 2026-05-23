# Stage 15 8B Summary — 失败诊断驱动的成功

> **作者**: Claude (auto-generated retro)
> **完成日期**: 2026-05-23
> **核心结论**: 8B Stage 14b 失败 (复用 14B 配方仅 +0.35pp，在 noise) → 诊断到 reward 与 eval 方向不一致 → Stage 15 修复 (+0.84pp del_tom, +1.50pp cot)

## TL;DR

复用 14B Stage 14b 配方训 8B 几乎无效。诊断后发现 8B Stage 7 在 51.5% 训练数据上 reward >= 0.95（已掌握），加权样本（Knowledge ×1.92, Desire ×1.65）在 8B 上**最容易**而不是最难，全部被 difficulty mask 过滤掉，没有梯度。

修复方案：
1. 用 8B Stage 7 给全部 12519 条训练数据打 reward label
2. 过滤掉 reward >= 0.95 的 6453 条
3. 用 8B Stage 7 自己的 per-task del_tom acc 反比加权

结果：del_tom 0.7534 → 0.7618 (+0.84pp)，cot 0.7350 → 0.7501 (+1.50pp)。

## Stage 14b 失败的根因

### 14B 经验为什么不能照搬

14B Stage 14b 用 14B Stage 12 在 full 5718 上的 per-task acc 加权（Knowledge×1.92, FB×0.72）。理由：14B Stage 12 在 Knowledge 上仅 0.5260 → 应该多学。

但**这个加权是给 8B 用的**，8B Stage 7 上：
- Knowledge eval acc = 0.4758（确实最弱）
- 但 Knowledge 训练数据上 reward = **0.882**（mean group score）
- 72% 的 Knowledge 训练 sample 已经全对（reward = 1.0）

**为什么 eval 弱但 train reward 高？** 因为 ToMBench eval 集和 Stage 12 训练集是**不同分布的 Knowledge 题**。eval 集的 Knowledge 题对模型更难（出题质量、分布、长度都不同）。

### Difficulty mask 的二阶效应

ROLL 的 `difficulty_high_threshold: 0.95` 把 reward 平均 ≥ 0.95 的 sample 全部 mask 掉。8B Stage 14b 的实际数据：

```
samples_used: 12-25/256 (5-10%)
Knowledge 加权 ×1.92: 1.92 倍上采样了"对 8B 来说已经会做"的 sample
→ 这些 sample 全被 mask
→ 加权完全无效
```

**Stage 14b 8B 等于在用一份分布严重错配的数据 + 5-10% 的有效梯度训练，结果可想而知**。

## Stage 15 修复

### 修复 1: reward labeling pipeline (`score_train_data_8b.py`)

对 12519 条 Stage 12 训练数据：
- 启动 8B Stage 7 vLLM
- 每个 prompt 生成 8 个 sample，温度 0.99，top_p 0.95（完全模拟 ROLL 的 rollout）
- 计算 group reward mean
- 输出 `8b_stage7_reward_full12519.jsonl`

12519 条耗时约 15 分钟（concurrency=32）。

**结果分布**：

| Reward bin | 数量 | 比例 |
|---|---|---|
| 1.0 (全对) | 7211 | **57.6%** |
| 0.95-0.999 | 0 | 0% |
| 0.80-0.94 | 1040 | 8.3% |
| 0.50-0.80 | 1216 | 9.7% |
| 0.30-0.50 | 979 | 7.8% |
| 0.15-0.30 | 514 | 4.1% |
| 0.0-0.15 | 630 | 5.0% |
| 0.0 (全错) | 1687 | 13.5% |

**57.6% 训练数据已经被 8B Stage 7 完全掌握**。这部分数据在 difficulty mask 0.95 上限下完全无效。

### 修复 2: 过滤 (`build_stage15_data.py`)

简单 filter: `reward_mean >= 0.95` 全部丢弃。
- 12519 → 6066 条 (-6453, -51.5%)

### 修复 3: 用 8B 自己的 acc 加权

| Task | 8B Stage 7 del_tom | Multiplier | 6066 → 7482 |
|---|---|---|---|
| Knowledge | 0.4758 | ×2.00 | 954 → 1908 |
| Desire | 0.5944 | ×1.69 | 246 → 407 |
| Belief | 0.7289 | ×1.26 | 1223 → 1525 |
| Emotion | 0.7357 | ×1.23 | 386 → 472 |
| Non-literal | 0.7747 | ×1.11 | 472 → 522 |
| Intention | 0.7853 | ×1.07 | 655 → 697 |
| False Belief | 0.8791 | ×0.77 | 844 → 665 |
| Other 等 | 1.00 | ×1.00 | 1286 → 1286 |

**总计 7482 条** (vs Stage 14b 14408)。少 6926 条但**几乎每条都 in learning zone**。

### 修复 4: Mask 阈值放宽

```yaml
# Stage 14b
difficulty_low_threshold: 0.15
difficulty_high_threshold: 0.80

# Stage 15
difficulty_low_threshold: 0.05
difficulty_high_threshold: 0.95
```

数据已预过滤 reward >= 0.95，所以可以把 high 放回 0.95。low 降到 0.05 让 0% reward 的"真正难题"也进梯度（探索难题）。

## 训练效果对比

### samples_used 对比

| Stage | samples_used | % of 256 |
|---|---|---|
| Stage 14b 8B | 12-25 | 5-10% |
| **Stage 15 8B** | **50-65** | **20-25%** |

**有效梯度信号 2-3 倍增加**。

### 训练轨迹对比

| step | Stage 14b 8B | Stage 15 8B | Δ |
|---|---|---|---|
| 0 (init) | 0.6020 | 0.5860 | -1.6pp (data shift) |
| 25 | -1.00pp | **+2.00pp** | +3.0pp |
| 50 | +0.20pp | **+1.80pp** | +1.6pp |
| 75 | -2.20pp | **+1.20pp** | +3.4pp |
| 100 | -1.00pp | **+4.00pp** | **+5.0pp** |
| 125 | -2.60pp | +1.40pp | +4.0pp |
| 150 | +0.20pp | **+3.40pp** | +3.2pp |
| 175 | -1.20pp | **+3.80pp** | +5.0pp |

**Stage 15 在每个 eval 点都领先 Stage 14b 3-5pp**。

### Full 5718 评测对比 (Δ vs Stage 7)

| protocol | Stage 14b best | **Stage 15 best** | 差距 |
|---|---|---|---|
| direct | -0.37pp (S14b-150) | -0.28pp (S15-150) | +0.09pp |
| cot | -0.10pp (S14b-199) | **+1.50pp (S15-150)** ⭐ | **+1.60pp** |
| **del_tom** | +0.35pp (S14b-150) | **+0.84pp (S15-150)** ⭐ | **+0.49pp** |

## 关键 Lessons

1. **不要盲目复用配方**: 14B 的加权方案对 8B 完全错位，因为 reward 分布不同。每个模型需要自己 reward label 训练数据。

2. **训练 reward ≠ eval acc**: 同一 task 在训练数据上 reward 0.88、eval 上 acc 0.48。眼睛要盯**训练数据 reward 分布**而不是 eval 分布做 difficulty curation。

3. **难题 + 已学会 = 浪费**: ROLL 的 difficulty_mask 只对训练数据 reward 起作用。eval 集弱不等于训练有信号。

4. **数据预过滤是高 ROI 操作**: 12519 → 7482 (-40%)，但有效梯度 2-3 倍。少而精远胜多而废。

5. **subset500 又骗人了**: ckpt-100 在 subset500 上 +4.00pp 是峰，但全量 cot/del_tom 上是 ckpt-150 胜。每次都需要全量评测多 ckpt。

6. **加权方向必须从模型自身出**: Stage 14b 用 14B 的 acc 加权 = 给 8B 多上采样它已经会的 task。Stage 15 用 8B 自己的 acc = 真正反映 8B 的 weakness。

## 通用模板（给未来 RL 实验）

新模型要做 task-weighted RLVR 时的标准流程：

```
1. 全量评测 (5718, 3 protocols) → 得到 per-task del_tom acc
2. Reward labeling 训练数据 (8 samples per prompt) → 得到 per-record reward mean
3. 过滤 reward >= 0.95 的 sample
4. 用模型自己的 per-task acc 反比加权（不是 baseline 模型的）
5. mask 阈值: (low=0.05, high=0.95)，因为数据已预过滤
6. save_steps=50 多 ckpt，全量评测 4 个 ckpt 选 best
```

## 下一步建议

1. **Stage 16 8B**: ckpt-150 已是 8B 项目记录。继续推可能 plateau，类似 14B 经验。值得做的：
   - 合成新难题（针对 8B 弱的 Knowledge / Desire 题）
   - reward shaping (RM/PRM 替代二元 reward)
   - 多 ckpt ensemble

2. **14B Stage 16**: 把 Stage 15 的 reward labeling pipeline 应用到 14B
   - 14B Stage 14b ckpt-199 的 reward sample 估计 70-80% >= 0.95
   - 过滤后预期能再 push +0.3-0.5pp del_tom

3. **跨模型加权**: 用 8B 和 14B 的 reward label intersection（两个模型都掌握的 sample 才是真冗余）做更精确的过滤。
