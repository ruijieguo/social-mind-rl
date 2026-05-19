# 改进方案 v2：算法 + 训练策略 + 测试时计算（不只是数据）

> 基于 stage 1-8 完整复盘 + 2025 年 RL/LLM 最新进展调研
>
> 目标: 14B 突破 raw 0.78 / clean 0.86, 8B 突破 raw 0.76 / clean 0.85, 在多个维度逼近 GPT-5.5

---

## 0. 当前瓶颈复盘

### 0.1 核心数据点

| 模型 | Raw 5718 | Clean 4551 | Subset500 best |
|---|---|---|---|
| 8B stage7 | 0.7419 | 0.8321 | 0.7480 (del_tom) |
| 14B stage8 | 0.7594 | 0.8449 | 0.7920 (del_tom, 反超 deepseek) |
| deepseek-v4-pro | 0.8080 | 0.9013 | 0.7880 |
| GPT-5.5 | 0.8349 | 0.9343 | — |

距 GPT-5.5: 8B raw -9.30pp / 14B raw -7.55pp

### 0.2 训练 saturation 的精确数据 (14B Stage8 step-by-step)

| step | rollout | samples_used/256 | all_correct | all_wrong | final_mask |
|---|---|---|---|---|---|
| 50 | 0.802 | 22% (57/256) | 0.44 | 0.06 | 0.22 |
| 100 | 0.815 | 13% (33/256) | 0.62 | 0.12 | 0.13 |
| **150** | **0.949** | **5% (14/256)** | **0.84** | 0.03 | **0.05** |
| 200 | 0.918 | 11% (28/256) | 0.81 | 0.03 | 0.11 |
| 349 | 0.906 | 6% (16/256) | 0.78 | 0.06 | 0.06 |

**致命问题**：step 150 后**只有 5-11% 训练样本贡献梯度**。其余 89-95% 被难度遮罩 (0.1/0.95) 丢弃，因为模型在它们上太好或太差。**模型在 reverse-saturating: 后期 200 步几乎不学新东西，仅在已会的样本上反复确认**。

### 0.3 三类未充分利用的"空间"

A. **饱和样本浪费**：89-95% 的 rollout 样本不产生梯度。这是**算法层面**问题，不是数据问题。

B. **测试时单次预测**：所有评测都是 temperature=0 单次解码。**未利用任何 test-time compute**。

C. **格式/长度奖励的副作用**：r_fmt × r_len 让模型学到"短答案 + boxed format"，但 Knowledge / Non-literal 这些需要 multi-step 推理的 task 反而被压制。

---

## 1. 算法改进（A 类瓶颈）

### A.1 [必做] Dr.GRPO — 修复 GRPO 长度偏差

**问题**: 标准 GRPO 把每个 token 的 advantage 用相同权重平均（"token aggregation bias"），导致：
- 错误回答倾向于更长（每个 token 只承担 1/L 的负 advantage，模型"懒得改")
- 我们的 r_total 是乘法，r_len 部分压制了这个偏差，但没消除

**Dr.GRPO 改动**:
```python
# 原 GRPO
advantage = (r - r_group_mean) / r_group_std
loss = -mean(advantage * log_prob_per_token)

# Dr.GRPO
advantage = r - r_group_mean       # 不除 std (避免偏差放大)
loss = -sum(advantage * log_prob_per_token) / max_len  # 不除 actual length
```

**预期效果**: 减小错误回答的长度偏置，对 Knowledge / Non-literal Comm 这些 trap-option 多的 task 帮助大（模型学得更"决断"）。

**实施**: 改 `framework/ROLL/.../grpo_loss.py` 三行代码。重训 stage8 → stage8b（350 步，9259 数据）。
**预计增益**: +0.5-1.0pp full eval

参考: [Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783), [Dr.GRPO topic](https://www.emergentmind.com/topics/dr-grpo)

### A.2 [必做] 难度遮罩动态调整

**问题**: 我们用固定 0.1/0.95 阈值。Step 50 还有 22% 样本贡献梯度（healthy），step 150 降到 5%（饥渴）。

**自适应难度遮罩**:
```python
# 根据 samples_used 比例动态调整阈值，目标保持 30%+ 贡献率
if samples_used_frac < 0.20:
    # 太多样本被丢，放宽阈值
    low_threshold = max(0.05, low_threshold - 0.02)
    high_threshold = min(0.99, high_threshold + 0.02)
elif samples_used_frac > 0.50:
    # 信号太多，可以收紧
    low_threshold = min(0.15, low_threshold + 0.01)
    high_threshold = max(0.90, high_threshold - 0.01)
```

**预期效果**: step 200+ 的 rollout 不浪费，模型继续学习接近 ceiling 的难题。

**实施**: 改 `framework/ROLL/roll/pipeline/rlvr/scheduler.py` 加 adaptive logic。
**预计增益**: +0.5-1.0pp（恢复后期训练有效性）

参考: [Curriculum Reinforcement Learning Easy to Hard](https://arxiv.org/abs/2506.06632), [Self-Evolving Curriculum for LLM Reasoning](https://arxiv.org/pdf/2505.14970)

### A.3 [极高 ROI] 渐进难度课程 (Easy → Medium → Hard)

**问题**: 当前训练数据**所有难度混杂**。Step 100 模型已能搞定 60% 题目，但训练继续给它喂混合难度，浪费 capacity。

**E2H Curriculum**:
1. **训练前**: 用 base model 在每条训练样本跑 8 次 sample，得到 success_rate
2. **分桶**: easy (>80%), medium (40-80%), hard (<40%)
3. **训练阶段**:
   - step 0-100: 50% easy + 40% medium + 10% hard（建立格式 + 中等推理）
   - step 100-200: 20% easy + 50% medium + 30% hard（深入推理）
   - step 200+: 5% easy + 35% medium + 60% hard（突破 ceiling）

**实施**:
1. `scripts/data/score_train_data.py`: 离线跑 base model 给每条数据打分
2. `scripts/data/build_curriculum.py`: 把 tom_train.jsonl 切成 3 个桶
3. ROLL data sampler 加 `curriculum_schedule` 参数

**预计增益**: 14B +1.5-2.5pp, 8B +1.0-1.5pp（突破 8B 0.74 ceiling）

参考: [E2H Curriculum (Learning Like Humans)](https://arxiv.org/abs/2505.08364)

### A.4 [实验性] VAPO — 替代 GRPO 用 value-based RL

**背景**: ByteDance 2025 论文 [VAPO](https://arxiv.org/html/2504.05118v1) 证明在 reasoning task 上，**value-based** 方法（PPO-style 带 critic）+ 长度自适应 GAE + value pretraining + 正样本 LM loss 综合效果优于 GRPO/DAPO。

**核心改动 vs GRPO**:
- 加一个 critic head（在 base model 之上）
- 用 GAE 算 advantage（比 group-relative 更稳定）
- "Length-Adaptive GAE": 不同 response 长度用不同 lambda
- "Value Pretraining": 第一阶段冻结 actor，只训 critic

**预期效果**: 训练更稳定，后期不会像 GRPO 那样 saturate。VAPO 在 AIME 2024 上比 DAPO +6pp。

**实施成本高**: 需要在 ROLL 里加 critic worker + 改 actor_train loss。投入 1-2 周开发。

**预计增益**: 14B +2-3pp（如果 ToMBench 的 RL 难度类似 AIME）

参考: [VAPO ByteDance 2025](https://arxiv.org/pdf/2504.05118), [llm-stats post-training 2026](https://llm-stats.com/blog/research/post-training-techniques-2026)

### A.5 [必做] KL=true 但用 token-level KL_loss 而非 token-level KL_ref

**问题**: stage1-8 都用 `add_token_level_kl: false`。原因是 stage3 试过 `add_token_level_kl: true` 反而性能下降——但那是因为 KL 系数太大。

**正确配置**:
```yaml
add_token_level_kl: true
kl_coef: 0.001  # 原 stage3 用了 0.01 太重；0.001 是 RLOO 推荐
kl_target: null  # 不用 adaptive target
```

**为什么需要**: 防止策略偏离 base model 太远。stage8 KL loss 累积到 0.42（很高），意味着策略已经远离 base 分布——这是 saturate + 过拟合的同时表现。

**预计增益**: +0.3-0.5pp（轻微但稳定）

---

## 2. 奖励改进（C 类瓶颈）

### B.1 [必做] 减弱 r_fmt × r_len 的乘法压制

**问题**: 当前 `r_total = r_fmt × r_out × r_len`（乘法）。一旦 r_fmt=0 或 r_len ≈ 0，整个奖励就是 0，模型完全不知道答案对错。

**改加权和**:
```python
r_total = 0.05 * r_fmt + 0.85 * r_out + 0.10 * r_len
# 或更激进
r_total = r_out * (0.5 + 0.5 * r_fmt) + 0.10 * r_len
```

**预期效果**: 答对了但格式略偏的样本不会被惩罚到 0，模型保留正确答案的 credit。Non-literal Comm 这种需要长解释的 task 受益最大。

**预计增益**: +0.3-0.6pp（特别是 Knowledge / Non-literal）

### B.2 [实验性] Process Reward Model (PRM) 替代 outcome reward

**当前**: 只用 outcome (`r_out` 0/1) 作为答案正确性奖励
**Process Reward**: 每个推理 step 给一个奖励（部分正确也得分）

**实施难度高**: 需要训一个 PRM (process reward model)。
**预计增益**: +0.5-1pp 但需要单独训练 PRM。

**优先级低**: 暂不推荐，除非 stage9 完成后还想榨剩下的 1pp。

---

## 3. 测试时计算（B 类瓶颈，**ROI 最高**）

### C.1 [必做，立即实施] Self-Consistency (Majority Voting)

**问题**: 我们所有 eval 都是 temperature=0 单次解码。GPT-5.5 / deepseek 内部用了 reasoning（test-time compute）才有那个分。

**Self-Consistency**:
1. 用 temperature=0.7 sample N=16 个答案
2. 取 majority vote 作为最终答案

**预期增益**（基于 [Self-Consistency 文献](https://arxiv.org/pdf/2511.12309)）:
- N=4 多数投票: 通常 +2-4pp
- N=16: +4-8pp
- N=32: +5-10pp（边际收益递减）

**对我们的具体预测**:
- 14B stage8: 0.7594 → ~0.79 (N=4) → ~0.81 (N=16) → ~0.83 (N=32)
- 8B stage7: 0.7419 → ~0.77 (N=4) → ~0.79 (N=16)

**N=16 时 14B 几乎能持平 deepseek-v4-pro 的 0.8080，N=32 接近 GPT-5.5 0.8349**！

**实施成本**: ~1 天工作。
1. 改 `scripts/eval/run_tombench.py` 加 `--self-consistency-n` 参数
2. 评测耗时 × N（14B 7min × 16 = ~2h，可接受）

**预计增益**: 14B +5-8pp on raw eval, +6-9pp on clean eval

参考: [Optimal Self-Consistency for Efficient Reasoning](https://arxiv.org/pdf/2511.12309), [Certified Self-Consistency](https://arxiv.org/pdf/2510.17472)

### C.2 [高 ROI] Confidence-aware sampling (DeepConf)

**改进 self-consistency**: 不是均匀 majority vote，而是用模型自己的 logit confidence 加权。

**算法**:
1. Sample N 个答案，每个答案附 log-prob
2. 答案 X 的得分 = sum(exp(log_prob)) over all samples that picked X
3. 取得分最高的 X

**预期效果**: 比 majority vote 高 +0.5-1pp

参考: [Deep Think with Confidence](https://jiaweizzhao.github.io/deepconf/static/pdfs/deepconf_arxiv.pdf)

### C.3 [必做] Multi-protocol ensemble

**当前**: 三个协议 direct/cot/del_tom 各跑各的。
**改进**: 同一题用三个协议各 sample 1 次，3 个答案 majority vote（如果一致则高置信，不一致则用 confidence weighting）。

**实施**: 改 `scripts/eval/run_tombench.py` 加 `--ensemble-protocols`。
**预计增益**: +1-2pp（额外）

### C.4 [必做] Best-of-N with critic / verifier

**进一步优化**: Sample N 答案后，让一个 verifier 模型选最好的，而不是 majority vote。
**实施**: 用 14B 自己作为 verifier（"是否正确？"二选一），或调 GPT-5.5 选。
**预计增益**: 比 majority vote 高 +0.5-1pp

---

## 4. 模型规模 + 蒸馏

### D.1 [终极方案] Qwen3-32B 训练

**预测**: 32B base ≈ 0.76, 32B + stage8 配方 ≈ 0.79-0.81 (raw), 0.86-0.87 (clean)
**资源需求**: 1×8 H800 TP=4 + DP=2，约 14h 训练。

### D.2 [新建议] GPT-5.5 蒸馏（替代 RL）

**思路**: 我们已经有 GPT-5.5 在 5718 题上的预测。把这些当作"金标准"做 SFT 蒸馏，而不是 RL。

**算法**:
1. 用 GPT-5.5 生成每条题目的正确答案 + 推理链 (CoT)
2. 用 SFT 教 14B 模型匹配 GPT-5.5 的推理过程
3. 后接 GRPO RL fine-tune

**预期效果**: SFT 部分能让模型学到 GPT-5.5 的推理结构（不只是答案），可能突破纯 RL 的 ceiling。

**实施**:
1. GPT-5.5 在 9259 训练数据上生成 reasoning traces (~$200, ~3h)
2. 用 LoRA SFT 14B base on traces (~4h)
3. 然后 stage9 GRPO RL on top

**预计增益**: +2-4pp（如果蒸馏成功）

参考: [Minimalist Approach (rejection sampling + RL)](https://arxiv.org/html/2504.11343v1)

### D.3 [新建议] Reject-Sampling Fine-Tuning (RFT) 简单基线

**思路**: 用 14B stage8 生成 N=16 答案/题，**只保留答对的回答**，组成 SFT 数据集，再 SFT。

**与 RL 区别**: 不需要 advantage 信号，只用正确性 filter。
**优点**: 训练稳定，无 saturate。

**实施**:
1. 14B stage8 vLLM serve, sample N=16 per question with T=0.7
2. Filter 答对的 (~30-50% 训练数据通过)
3. SFT 14B base 在 filtered 数据上

**预计增益**: 与 GRPO 相当，但训练稳定。可作为 GRPO 的 hot-start。

参考: [Predibase: RL beats SFT with limited data](https://predibase.com/blog/how-reinforcement-learning-beats-supervised-fine-tuning-when-data-is-scarce)

---

## 5. 综合方案（优先级排序 by ROI）

### Phase 1（1 周内可完成，预计 +6-10pp on full 5718 raw）

**🎯 P1: Self-Consistency N=16** [必做] 
- 实施: ~1 天
- 预计增益: 14B +5-8pp, 8B +4-6pp
- 成本: 评测时间 × 16
- **直接突破 deepseek 0.8080**

**🎯 P2: Multi-protocol ensemble** [必做]
- 实施: ~1 天
- 预计增益: 额外 +1-2pp
- 成本: 评测时间 × 3 协议

**🎯 P3: Dr.GRPO loss 修复** [必做]
- 实施: 改 3 行代码 + 重训 stage8b（350 步，~7h）
- 预计增益: +0.5-1pp on raw

合计 Phase 1 14B raw: **0.7594 → 0.81-0.84**（持平 deepseek，接近 GPT-5.5）

### Phase 2（2 周内可完成，预计再 +2-4pp）

**🎯 P4: 难度课程 (E2H)** [极高 ROI]
- 实施: 1) score data ~3h, 2) 改 sampler ~1 天, 3) 重训 stage9（350 步，~7h）
- 预计增益: 14B +1.5-2.5pp, 8B +1-1.5pp（突破 8B ceiling）

**🎯 P5: 自适应难度遮罩** [必做]
- 实施: 改 ROLL scheduler ~半天
- 预计增益: +0.5-1pp（恢复后期训练有效性）

**🎯 P6: 加权和奖励 (替代乘法)** [必做]
- 实施: 改 reward worker 1 行 + 重训
- 预计增益: +0.3-0.6pp

**🎯 P7: KL=true with kl_coef=0.001**
- 实施: 改 config 1 行 + 重训
- 预计增益: +0.3-0.5pp

### Phase 3（1 个月内可完成，预计再 +3-5pp）

**🎯 P8: GPT-5.5 蒸馏 + RL hot-start** [最高 ROI]
- 实施: 蒸馏 + SFT + GRPO ~1 周
- 预计增益: +2-4pp

**🎯 P9: VAPO 替代 GRPO**
- 实施: ~2 周
- 预计增益: +1-2pp（或更多，未知）

**🎯 P10: 32B 模型训练**
- 资源: 1×8 H800, 14h
- 预计增益: +2-3pp（vs 14B）

---

## 6. 推荐立即执行的 minimal action plan

**Week 1**: P1 + P2 + P3
- Self-consistency + multi-protocol + Dr.GRPO 重训
- **预期最终: 14B raw 0.81-0.84, clean 0.87-0.90**

**Week 2-3**: P4 + P5 + P6
- 难度课程 + 自适应遮罩 + 加权和奖励
- **预期最终: 14B raw 0.84-0.86, clean 0.90-0.92**（持平 deepseek-v4-pro，接近 GPT-5.5）

**Week 4+**: P8（GPT-5.5 蒸馏） 或 P10（32B）二选一

---

## 7. 各方案预期增益 vs 成本对比

| 方案 | 预计增益 (raw) | 实施成本 | 训练时间 | API/GPU 成本 | 风险 | ROI |
|---|---|---|---|---|---|---|
| **P1 Self-Consistency N=16** | **+5-8pp** | 1 天 | 0 | $0 (本地 vLLM) | 低 | **⭐⭐⭐⭐⭐** |
| P2 Multi-protocol ensemble | +1-2pp | 1 天 | 0 | $0 | 低 | ⭐⭐⭐⭐ |
| P3 Dr.GRPO | +0.5-1pp | 1 天 | 7h | 56 GPU-h | 低 | ⭐⭐⭐ |
| **P4 难度课程 (E2H)** | **+1.5-2.5pp** | 2 天 | 7h | 56 GPU-h | 低 | **⭐⭐⭐⭐⭐** |
| P5 自适应遮罩 | +0.5-1pp | 0.5 天 | 7h | 56 GPU-h | 低 | ⭐⭐⭐⭐ |
| P6 加权和奖励 | +0.3-0.6pp | 0.5 天 | 7h | 56 GPU-h | 低 | ⭐⭐⭐ |
| P7 KL=true 微调 | +0.3-0.5pp | 0.1 天 | 7h | 56 GPU-h | 低 | ⭐⭐ |
| **P8 GPT-5.5 蒸馏** | **+2-4pp** | 5 天 | 12h | $200 + 96 GPU-h | 中 | **⭐⭐⭐⭐⭐** |
| P9 VAPO | +1-2pp | 14 天 | 8h | 64 GPU-h | 高 | ⭐⭐⭐ |
| P10 32B | +2-3pp | 1 天 | 14h | 112 GPU-h | 中 | ⭐⭐⭐⭐ |

**最高 ROI 的三件事**: P1 (Self-Consistency) + P4 (难度课程) + P8 (GPT-5.5 蒸馏)

合计预期: 14B 从 0.7594 推到 **0.85-0.88 (raw) / 0.91-0.94 (clean)**，持平 GPT-5.5 raw 或反超。

---

## 8. 与之前 improvement_plan.md 的关键差异

| 维度 | improvement_plan.md (老版) | 本方案 (新版) |
|---|---|---|
| 主要思路 | 数据扩展 (Phase A/B/C) | **算法 + Test-time compute + 蒸馏** |
| 数据 | 加 1500-2300 条新数据 | 数据已基本到位，主要清洁/重组 |
| 算法改动 | 难度课程（提到但未实施） | **Dr.GRPO + 自适应遮罩 + VAPO + KL 修复** |
| Test-time | 完全没考虑 | **Self-consistency N=16 + ensemble + DeepConf** |
| 蒸馏 | 完全没考虑 | **GPT-5.5 reasoning trace 蒸馏 + RL** |
| 预期增益 | 14B +2-3pp | **14B +6-12pp（综合方案）** |

**老方案的局限**: 只关注训练数据本身，没意识到我们 95% rollout 浪费 + 0% test-time compute 的双重问题。

**新方案的核心 insight**: 
1. 我们的训练数据已经接近最优（Phase C 风格匹配 + Phase B 中文 + base 7259 cleaned）
2. 真正的"免费午餐"在 test-time compute 和算法层
3. self-consistency N=16 一夜之间能 +5-8pp，比 1 个月再造一批数据收益高

---

## 9. 结论

**未来 14B 突破 0.85 / 8B 突破 0.78 的最短路径**:

1. **本周**: 实施 P1 (Self-Consistency N=16) — **预期立即获得 +5-8pp**
2. **下周**: 加 P2 (multi-protocol) + P3 (Dr.GRPO) — **再 +1-2pp**
3. **第三周**: 实施 P4 (难度课程) — **+1.5-2.5pp**
4. **第四周**: P8 (GPT-5.5 蒸馏) — **+2-4pp**

**累计预期**: 14B 从 0.7594 → **0.85-0.88 (raw) / 0.91-0.94 (clean)**。

**关键认识**: 
- 我们之前 7 个 stage 都在数据维度兜兜转转，从 stage1 0.7527 到 stage8 0.7594 仅 +0.67pp
- 算法 + test-time compute 一夜之间能多 +5-10pp
- **这个项目从"数据 bound"转移到"算法 + 推理 bound"**

最后更新: 2026-05-19 16:30

## 引用文献

- [VAPO (ByteDance 2025)](https://arxiv.org/pdf/2504.05118): Value-based RL beats GRPO/DAPO on AIME +6pp
- [Dr.GRPO (COLM 2025)](https://arxiv.org/pdf/2503.20783): GRPO 长度偏差修复
- [E2H Curriculum (EMNLP 2025)](https://arxiv.org/abs/2505.08364): Easy-to-Hard 课程提升 small LLM
- [Self-Consistency Optimal Sampling](https://arxiv.org/pdf/2511.12309): N 样本最优数量分析
- [DeepConf (2025)](https://jiaweizzhao.github.io/deepconf/static/pdfs/deepconf_arxiv.pdf): Confidence-weighted majority voting
- [Curriculum RL Easy-to-Hard (2025)](https://arxiv.org/abs/2506.06632): RL 难度课程
- [Self-Evolving Curriculum](https://arxiv.org/pdf/2505.14970): 自适应课程
- [Minimalist RL (RFT)](https://arxiv.org/html/2504.11343v1): Rejection sampling 替代 RL
- [Post-Training in 2026](https://llm-stats.com/blog/research/post-training-techniques-2026): 综述
