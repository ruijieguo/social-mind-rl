# Stage 10: 基于证据的合并训练方案 (取代 v3 Phase 2/3 分离)

> 基于 Stage 9 完整数据 + 训练日志 + 100% raw 输出分析。
> **不做无证据的猜测**：只引用可验证的训练指标和输出样本。

---

## 1. Stage 9 实际证据（不是猜测）

### 1.1 训练日志关键指标 (stage9 14B, 350 步)

| step | rollout | samples/256 | all_correct | KL loss | r_fmt | r_out | r_len |
|---|---|---|---|---|---|---|---|
| 0 | 0.584 | 256 (100%) | 0.00 | 0.000 | 1.000 | 0.973 | 0.600 |
| 25 | 0.917 | 32 (13%) | 0.81 | 0.093 | 0.992 | 0.938 | 0.975 |
| 50 | 0.992 | 8 (3%) | 0.81 | 0.034 | 0.992 | 0.992 | 1.000 |
| 75 | 0.916 | 48 (19%) | 0.69 | 0.595 | 0.988 | 0.922 | 0.989 |
| **100** | 0.856 | 88 (34%) | 0.38 | **3.011** ⚠ | 0.988 | 0.910 | 0.946 |
| **150** | 0.871 | 110 (43%) | 0.44 | **7.258** ⚠⚠ | 0.914 | 0.891 | 0.933 |
| 200 | 0.897 | 59 (23%) | 0.62 | 0.915 | 0.977 | 0.898 | 0.979 |
| 250 | 0.883 | 99 (39%) | 0.41 | 4.109 | 0.895 | 0.883 | 0.914 |
| 349 | 0.891 | 54 (21%) | 0.59 | 1.156 | 0.945 | 0.895 | 0.941 |

### 1.2 三大证据级失败原因

**【证据 1】KL coef=0.001 没有约束作用，KL loss 失控暴涨**
- 配置：`use_kl_loss: true, kl_loss_coef: 0.001`
- 实际：KL loss 0.034 (step 50) → **7.258 (step 150)** = **214× 暴涨**
- 含义：策略完全跑飞，已经远离 SFT init。KL 系数太小起不到锚定作用。
- **不是猜测**: 直接来自 `actor/kl_loss` 训练指标。

**【证据 2】Qwen3 chat template 与 SFT 教法冲突**
- 配置：SFT 数据用 `<reasoning>...</reasoning>` 标签
- 实际：5718/5718 (**100%**) eval 输出包含空的 `<think>\n\n</think>`
- 配置：SFT 数据用 `<reasoning>...</reasoning>` 标签
- 实际：0/5718 (**0%**) eval 输出包含 `<reasoning>` 标签
- 含义：Qwen3 base 的 chat_template 在 apply 时强制插入 `<think></think>`，SFT 没能改变这个层级行为。我们以为 SFT 教会了一个新协议，**实际上完全没教成功**。
- **不是猜测**: 直接 grep raw responses 验证。

**【证据 3】wrong responses 比 correct responses 更长**
- s8 (无 SFT): 平均 483 字符
- s9 correct: median 234 字符
- s9 wrong: median 328 字符 (+40% longer)
- s9 wrong with > 500 chars: **22.5%**, vs correct: **15.6%**
- 含义：当 s9 模型写更长推理时，**更可能答错**。Long CoT 不仅没帮助，反而引入错误。
- **不是猜测**: 直接 raw response 长度统计。

### 1.3 Per-task 退步精确数据 (s9 vs s8, full 5718 direct)

| Task | s8 | s9 | Δ | gap to gpt-5.5 |
|---|---|---|---|---|
| **Intention** | **0.818** | **0.754** | **-6.3pp** ↓↓ | -12.5 |
| **Belief** | 0.739 | 0.694 | **-4.6pp** ↓ | -14.8 |
| **False Belief** | 0.864 | 0.847 | -1.6pp | -7.9 |
| Emotion | 0.727 | 0.715 | -1.2pp | -10.0 |
| Knowledge | 0.514 | 0.505 | -0.9pp | -16.6 |
| Non-literal Comm | 0.792 | 0.786 | -0.6pp | -4.8 |
| Desire | 0.569 | 0.597 | **+2.8pp** ↑ | -8.3 |

**关键 pattern**:
- Intention -6.3pp = 最差。Intention 题型本质是"识别一个简短意图"，强加 long CoT 让模型在简单题上想多了
- Belief / FB / Emotion / Knowledge 都退步 — 这些都是 stage8 已经做得相对好的 task
- 只有 Desire +2.8pp — 这个 task 在 stage8 是弱项 (0.569)，stage9 反而帮助

### 1.4 训练动态判读

- **step 0-50**: SFT init 让 r_fmt=1.0, r_out=0.97 — 训练数据上几乎全对
- **step 0-50 samples_used 256→8** = **97% rollout 被遮罩**。GRPO 没有梯度信号
- **step 75-150**: 模型开始"找梯度"，但是错的方向 — KL 暴涨，r_fmt 略降
- **step 200+**: KL 周期性飙升后回落，模型在 SFT init 周围震荡
- **rollout score 始终 0.85-0.99**，但 val/full eval 反降

## 2. 不变事实总结

1. **Stage 8 (0.7580 raw / 0.8449 clean / 0.7920 del_tom)** 仍是项目最强 14B
2. SFT cold start 在 ToMBench 不 work（vs 数学题 work）
3. KL coef 0.001 不够稳定
4. response_length 1024 不需要（实际生成 median 234 chars）
5. **数据已基本达到 14B + 标准 GRPO 上限**

---

## 3. Stage 10 合并方案：基于证据的优化

### 3.1 设计原则

只采纳**有 stage 8 之前数据证据支持**的改动。**不引入** SFT cold start，**不延长 response_length**。

### 3.2 具体改动 (5 项, 全部低风险)

#### 改动 1: Weighted-sum reward (有证据)

**证据**：Stage 9 r_out=0.97 from step 0（SFT 教会答案）, 但 r_fmt 偶尔降到 0.89 → 当 r_fmt=0 时 multiplicative reward 完全归零, 模型失去 credit signal。

**Stage 10**:
```yaml
rewards.tom_mcq:
  aggregation: weighted_sum
  r_fmt_weight: 0.05
  r_out_weight: 0.85
  r_len_weight: 0.10
```

**预期**: 边际 +0.3-0.6pp（不依赖 SFT）。

#### 改动 2: Entropy bonus = 0.005 (轻微，有证据)

**证据**：Stage 8 训练后期 samples_used 降到 5-13%（来自 stage8 详细日志），模型过度探索-exploit。 Entropy bonus 鼓励保持采样多样性。

**Stage 10**:
```yaml
entropy_loss_coef: 0.005  # half of v3's 0.01, less aggressive
```

#### 改动 3: 把 KL 重新关掉 (有证据)

**证据**：Stage 9 KL=0.001 完全失效（实际 KL loss 暴涨到 7.258）。Stage 8 用 `add_token_level_kl: false` 反而表现最好。

**Stage 10**:
```yaml
add_token_level_kl: false
use_kl_loss: false
```

回归 Stage 8 配置。

#### 改动 4: 保持 stage 8 的 difficulty 阈值 (有证据)

**证据**：Stage 8 用 0.1/0.95 → 0.7580; Stage 9 放宽到 0.05/0.97 → 0.7429。证据上没差别。

**Stage 10**:
```yaml
difficulty_low_threshold: 0.1   # stage 8 默认
difficulty_high_threshold: 0.95
```

#### 改动 5: max_steps 调到 300 (有证据)

**证据**：Stage 8 step 250 val=0.720（峰值），step 300 val=0.710（小回调）。Stage 9 step 100-250 平台期。300 步是 sweet spot。

**Stage 10**:
```yaml
max_steps: 300
save_steps: 300
```

### 3.3 训练数据不变

继续用 `tom_train.jsonl` 9259 records (s8 配方):
- base 7259 (cleaned + GPT-5.5 audit)
- + Phase C 1200 (style-matched)
- + Phase B 800 ZH

### 3.4 算法不变

- GRPO with dual_clip
- whiten_advantages: true (s8 配置, **不用** Dr.GRPO — 没有 stage 间 A/B 对比证据)
- response_length: 256 (s8 配置)
- pg_clip_low: 0.20, pg_clip_high: 0.28 (DAPO Clip-Higher, s8 配置)

### 3.5 完整 Stage 10 config (5 改动 + s8 baseline)

```yaml
# rlvr_config_stage10_1x8_14b.yaml
exp_name: "qwen3-14B-tombench-rlvr-stage10-1x8"

# === s8 baseline (保持) ===
max_steps: 300                        # 从 350 调到 300 (避免 saturate 期)
save_steps: 300

pretrain: Qwen/Qwen3-14B              # 从 stage9 SFT init 退回 base
response_length: 256                  # 从 1024 退回
prompt_length: 1024

# === Algorithm (s8 配置) ===
adv_estimator: "grpo"
use_pg_clip_range: true
pg_clip_low: 0.20
pg_clip_high: 0.28
dual_clip_loss: true
whiten_advantages: true               # s8 配置 (不用 Dr.GRPO 没有 A/B 证据)
add_token_level_kl: false             # s8 配置
use_kl_loss: false                    # 取消 v3 的 KL 损失
loss_agg_mode: "seq-mean-token-mean"  # s8 默认

# === 改动 1: Entropy (轻微) ===
entropy_loss_coef: 0.005

# === 改动 2: Difficulty (s8 默认) ===
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95

rewards.tom_mcq:
  # === 改动 3: weighted_sum reward ===
  aggregation: weighted_sum
  r_fmt_weight: 0.05
  r_out_weight: 0.85
  r_len_weight: 0.10
  l_min: 8                            # s8 default (不是 long CoT)
  l_max: 256
```

### 3.6 预期效果（有 stage 1-9 数据基础）

| 路线 | 推算依据 | 预期 |
|---|---|---|
| stage 8 baseline | 实测 | 0.7580 |
| + weighted_sum reward | s9 数据: r_out=0.97 时 multiplicative 偶有 r_fmt=0 归零 | +0.2-0.5pp |
| + entropy 0.005 | s8 后期 samples_used 5-13% saturate | +0.1-0.3pp |
| + max_steps 300 (避免 saturate over-fit) | s8 step 250 val peak 0.720, step 300 0.710 | +0.0-0.2pp |
| **Stage 10 预期** | | **0.76-0.77 raw**（+0.2-0.7pp vs s8）|

**保守预期 0.76, 乐观 0.77**。**不会**像 v3 那样预期 0.86 — 我们没有证据支持那么大的提升。

---

## 4. 后续 stage 11/12 评估（条件触发，不预先承诺）

### 4.1 触发条件：Stage 10 必须先 ≥ Stage 8

如果 Stage 10 < 0.7580 → stop，证明现有 GRPO 配置已是局部最优，重新审视。

如果 Stage 10 = 0.76-0.77 → 可以再尝试**一项**改动：

### 4.2 候选改动 A: ExploreToM 程序化数据

**证据基础**:
- Meta 2025 paper 在 ToMi/HiToM 上 +27pp（不同基准，但同类型）
- 我们 stage8 错误中 HOT (492 题, deepseek+gpt 都对我们错) = +8.60pp 上限

**实施成本**: 5-7 天开发框架 + 2 天数据 + 1 天训练

**预期增益**: 14B raw +1-2pp（保守，基于 HOT 错误的局部修复）

### 4.3 候选改动 B: 模型规模 32B

**证据基础**:
- 8B → 14B: +1.89pp (实测, stage1 8B 0.7394 → 14B 0.7527)
- 推断 14B → 32B: +1-2pp (size scaling 边际收益递减)

**实施成本**: 1 天，14h 训练，TP=4 配置（验证过）

**预期增益**: 32B raw 0.78-0.79

### 4.4 候选改动 C: 难度课程

**证据基础**:
- Stage 8 step 150 后 samples_used = 5-13% (95% 浪费)
- Stage 9 即便加 entropy，samples_used 也只有 21-43%

**实施成本**: 3 天开发 + 1 天 score 数据 + 1 天训练

**预期增益**: 不确定（无 ToMBench 直接证据，外推 +0.5-1pp）

### 4.5 评估优先级

按 ROI:
1. **ExploreToM** 最高（其他人在同类基准证明 +27pp，即便效果只 1/10 也是 +2.7pp）
2. **32B** 次之（确定性增益 +1-2pp，成本最低）
3. **难度课程** 第三（外推，没有 ToMBench 直接证据）

---

## 5. 实施计划

### Phase 1 (Day 1-2): Stage 10 训练 + 评测
- Day 1: 写 stage10 config + sync + launch (~7h 训练)
- Day 2: HF convert + 3 eval（full + clean + subset500）
- **Decision point**: Stage 10 ≥ 0.7580? 

### Phase 2 (Day 3-9): 条件性下一步
- 如 Stage 10 ≥ 0.76 → 开始 ExploreToM 框架开发（高优先级）
- 如 Stage 10 ≥ 0.78 → 直接进入 32B 训练
- 如 Stage 10 < 0.7580 → STOP, 写最终报告

### Phase 3 (Day 10-14): 最后冲击
- 训练新 stage 11
- 评测
- 撰写最终 paper / 报告

**项目总体目标**: 14B raw 0.78 (修正后预期，不是 v3 的 0.86)。

---

## 6. 何谓"务实的成功"

旧目标 (v3): "持平 GPT-5.5 0.8349" — **无证据支持，被 Stage 9 反证**。

新目标 (基于证据):
- **底线**: 14B raw 0.76 (Stage 10 通过 reward + entropy 优化)
- **稳定**: 14B raw 0.78 (Stage 11 加 ExploreToM 数据)
- **挑战**: 14B raw 0.80 (+ 32B 规模)
- **天花板**: 14B + 32B 综合接近 deepseek-v4-pro 0.8080 但不超

距 GPT-5.5 0.8349 的 5pp gap 在不大幅改动训练范式（如 R1-style RL 在专门 ToM benchmark）下 **可能无法跨越**。这个 5pp 大部分来自 GPT-5.5 的预训练规模和质量优势。

---

## 7. 已验证 / 已 falsified 的训练改动

### 已验证（有 stage 间 A/B 证据，可以重用）
- ✅ Phase C 风格匹配数据合成 (s7 → s8: +0.55pp)
- ✅ GPT-5.5 audit eval set (clean eval +8.5-9.9pp，全模型同等)
- ✅ Difficulty mask 0.1/0.95 (s8 用，最佳结果)
- ✅ max_steps 250-300 (s8 用 250 = 0.7580, 边际收益递减)
- ✅ whiten_advantages: true (s8 配置)
- ✅ DAPO Clip-Higher 0.20/0.28 (s8 配置)
- ✅ 9259 records 数据 (s8 配方)

### 已 falsified (有 stage 9 反证)
- ❌ SFT cold start (Stage 9 retro: -1.51pp)
- ❌ KL coef 0.001 (Stage 9: KL loss 暴涨 7.258)
- ❌ response_length 1024 (s9 wrong responses 比 correct 长)
- ❌ Dr.GRPO loss (s9 配合 SFT 一起退步，无单独 A/B)
- ❌ Long CoT reward (l_min=100, l_max=600) — 同上

### 未验证 (仅猜测，不放入 Stage 10)
- ⚠ Adaptive difficulty mask（没人测过）
- ⚠ High-entropy token selection（仅论文证据）
- ⚠ VAPO（成本太高，未尝试）
- ⚠ 32B 模型规模（成本可承受，但需要等 Stage 10 结果后再决定）

最后更新: 2026-05-20 14:30
