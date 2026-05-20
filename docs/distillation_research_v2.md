# Stage 11 优化方案 (v2): 基于深度调研 + 训练经验教训

> 在 docs/distillation_research.md (v1) 基础上做更深度调研。
> 关键新发现：方案 A (RAFT self-distillation) 风险被低估。需要更精细的设计。

---

## 1. v1 方案 A 的隐患 (新证据)

### 1.1 Stage 8 在训练数据上已经太准 (新数据)

从 stage 8 训练日志 (`logs/train_stage8_1x8_14b_20260518_234107.log`) 实测：

| step | rollout score | r_out_mean | all_correct_groups | all_wrong_groups |
|---|---|---|---|---|
| 50 | 0.80 | 0.81 | 0.44 | 0.06 |
| 100 | 0.81 | 0.82 | 0.62 | 0.12 |
| 150 | 0.95 | 0.95 | **0.84** | 0.03 |
| 200 | 0.92 | 0.92 | 0.81 | 0.03 |
| 349 (final) | — | 0.90 | 0.78 | — |

**Stage 8 训练结束时**:
- 9259 题中约 **78% 的题，8/8 rollout (T=0.99) 全对**
- ~3% 的题，8/8 全错
- 中间 19% 是 partial（既有对又有错）

**RAFT self-distillation 在这种状态下的问题**:

1. **数据多样性低**: 78% 题 student 已经答得"完全正确"。重训这些 ≈ 直接复读训练集，**信息量极低**。
2. **真正能学的只有 19% partial**: 这些题学生有时对、有时错。可学习的真"自演化"信号只在这里。
3. **3% all-wrong 完全无救**: RAFT 直接抛弃这些，但**这正是 +1-2pp 增益空间所在**。
4. **Spurious correct 风险**: 78% 答对的题里，多少是"理由错但答案对"？ToMBench 标签有 40% 噪声 (s8 audit 数据)，即使学生 8/8 答对，可能因为"碰对标签错误的题"，再训反而强化偏差。

参考: [Beyond Rejection Sampling (2025)](https://arxiv.org/html/2602.04391) 明确警示 "an LLM trained on this spurious RFT data performs worse than the π_sft policy itself"。

### 1.2 我们 Stage 9 失败的核心，方案 A 没完全避免

Stage 9 失败原因：**off-policy 蒸馏 → 分布漂移**。
方案 A 是 **on-policy + gold-filter SFT**。但有新隐患：

- 学生 SFT 自己的 correct rollouts → **从 RL 策略退化回 SFT 策略**
- Stage 8 的策略是通过 GRPO 优化的（with advantage signals），其中包含**对错对比信号**
- 改用 SFT，丢失"对错相对"的信息，只保留"对"的信息
- 数学等价于：**降低了 GRPO 的探索/利用平衡，让模型变 narrower**

参考: [OPO (On-Policy RL with Optimal Reward Baseline, 2025)](https://arxiv.org/pdf/2505.23585): "GRPO 有时输出比 SFT 还低的 pass@16，OPO 改进 baseline 后才稳定"。这说明 **SFT-after-GRPO 是有反向风险的**。

### 1.3 Thinking Machines Lab 的 OPD 视角 (最关键新证据)

[Thinking Machines OPD blog](https://thinkingmachines.ai/blog/on-policy-distillation/) 实测：
- **On-policy distillation 用 per-token reverse KL** 训练学生
- 学生 sample 自己的 trajectory，teacher 对**每个 token** 给出 distribution 评分
- **Dense token-level supervision** > 稀疏 outcome reward
- 在 Qwen3 上**比 pure RL 用 fraction of compute** 达到同样性能

**问题**: 这需要 teacher logits — **GPT-5.5 API 没有**。我们用不了 OPD 的 reverse-KL 版本。

但 **OPD 的精神可用**：用 student 自己的 trajectory，给 dense supervision（不只是 outcome）。

### 1.4 IPO/DPO 视角 (新候选)

[It Takes Two: Your GRPO Is Secretly DPO (arxiv 2510.00977)](https://arxiv.org/html/2510.00977v3): 在 GRPO 中，**当所有 rollout 同 reward 时，sample 被 wasted**。这正是 stage 8 后期 samples_used 降到 14/256 (6%) 的原因。

[A-IPO (2025)](https://arxiv.org/html/2510.10077v1) + [Evaluating DPO variants (ACL 2025)](https://aclanthology.org/2025.acl-srw.26.pdf): DPO/IPO 用**对错配对**，比 GRPO 在 sparse-reward 后期更有效。

**这给了方案 A 一个升级路径**: 不只 SFT correct rollouts，而是构建 **(correct, wrong) preference pairs** 做 DPO/IPO training。

---

## 2. 方案 A 不会单独 work 的几个理由 (汇总)

| 因子 | 证据 | 影响 |
|---|---|---|
| 数据 80% 已答对 | s8 train log all_correct=0.78 | SFT 重训这些 = noise |
| 关键 wrong cases 被丢弃 | RAFT 只保留 correct | -3% all-wrong 无救，正是 ceiling |
| Spurious correct on noisy labels | s8 audit: 40% 错误是 wrong-label | 答对 ≠ 推理对 |
| SFT-after-GRPO 风险 | [Beyond Rejection Sampling 2025] | 可能比 stage 8 更差 |
| 同 task 重训倾向 narrow | [LLM catastrophic forgetting CSUR 2025] | hurt OOD eval |

**结论**: 单纯方案 A 预期 +1-2pp 是**乐观估计**。**0pp 或 -1pp 也完全可能**。

---

## 3. 改进的 Stage 11 方案 (基于深度调研)

### 3.1 设计原则 (针对每个风险)

| 风险 | v1 方案 A 应对 | v2 升级应对 |
|---|---|---|
| 数据 80% 已对 | 全部 SFT | **只 SFT partial-correct 题** (19% = ~1800 题) |
| Spurious correct | 信任 gold | **同 prompt 多 sample, 长度/格式 filter, 排除 luck-correct** |
| 抛弃 wrong cases | 丢 3% all-wrong | **DPO/IPO 用 (correct, wrong) pairs**, 包含 partial 题 |
| SFT-after-GRPO | 单纯 SFT | **极低 lr + 短训 + ckpt-on-val** |
| 同 task narrow | 1 epoch SFT | **数据多样化: 加 GPT-5.5 短 CoT for wrong, 加 OOD val 监控** |

### 3.2 升级方案 A+: DPO on (correct, wrong) pairs from student

**思路**: 不做 SFT, 做 **DPO 对比学习**。

1. Stage 8 14B sample N=8 / 题 (T=0.7) 在 9259 训练题上
2. 分类:
   - 8/8 all correct (~78%, 7200 题): **跳过**, 没有学习信号
   - 8/8 all wrong (~3%, 280 题): **跳过**, 学生学不会
   - **partial (~19%, 1780 题)**: **保留**, 这才是金矿
3. 在 partial 题中, 每题取 1 correct + 1 wrong → 形成 preference pair
4. **DPO 训练** (init from stage 8, lr 5e-7, 1 epoch)
5. 评测

**为什么这优于方案 A SFT**:
- ✅ **保留对错对比信号** (DPO 核心) — 不退化回 SFT
- ✅ **只训学生有摇摆的题** — 不浪费 80% 数据
- ✅ **不需要 teacher** — student own rollouts
- ✅ **理论保证**: DPO 收敛到隐式 reward model 最优
- ✅ **避免 spurious correct**: 必须有 wrong counterpart 才形成 pair, 强化"对错差"

**预期增益** (基于 [DPO variants ACL 2025](https://aclanthology.org/2025.acl-srw.26.pdf), [GRPO=DPO 等价 arxiv 2510.00977](https://arxiv.org/html/2510.00977v3)):
- DPO on 1780 partial pairs: 1 epoch ~30 min
- 预期 +0.5-1.5pp on raw eval (DPO literature typical)

### 3.3 改进方案 B+: GPT-5.5 short CoT 仅对 all-wrong 题

**只对学生 8/8 全错的 ~280 题** 让 GPT-5.5 生成短 CoT (≤200 chars, ≤3 steps)。

**为什么这好**:
- ✅ 280 题 × $0.05 = **$14** (vs v1 方案 B 的 $80, 因为只补 all-wrong 而非所有 wrong)
- ✅ Student 完全不会的题 → GPT-5.5 必须教
- ✅ Partial 题已经被方案 A+ 的 DPO 处理
- ✅ 短 CoT, no tag, inline reasoning → 避开 stage 9 陷阱

### 3.4 三阶段管线 (推荐最终方案)

**Phase 1**: Stage 8 rollout sampling
- vLLM serve stage 8, T=0.7, N=8 per prompt
- 9259 × 8 = 73k samples (~2-3h on 1 GPU)
- 分桶: all-correct, all-wrong, partial
- 输出: `data/stage11_rollouts.jsonl`

**Phase 2**: DPO on partial pairs (主菜)
- ~1800 partial 题 × 1 (correct, wrong) pair
- 训练 DPO 1 epoch, lr 5e-7 (init stage 8)
- 输出: stage 11a model
- 预计 +0.5-1.5pp

**Phase 3 (条件)**: SFT 补 all-wrong with GPT-5.5 short CoT
- 仅 ~280 题, $14 GPT-5.5 调用
- SFT 在 stage 11a 之上 1 epoch, lr 5e-7
- 输出: stage 11b model
- 预计 +0.5-1.5pp on top of 11a

**Phase 4**: Eval pipeline
- Full 5718 + clean 4551 + subset500 三协议
- 与 stage 8 (0.7594/0.8449/0.7920) 严格对比

### 3.5 关键防线: 不破坏 stage 8

每个 phase 都用 stage 8 ckpt as init (不是 stage 11a → 11b 串联训练)。这样：

- **Phase 2 失败** → 直接回 stage 8
- **Phase 3 失败** → 回 phase 2 或 stage 8

**catastrophic forgetting 缓解** ([CSUR 2025 LLM continual learning survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)):
- LR 5e-7 (vs base 1e-6, 减半)
- 1 epoch 严格上限
- Warmup 30 steps (vs default 100, 短训不必长 warmup)
- 监控 hold-out subset500 val acc, val drop > 0.5pp 就停

---

## 4. ROLL 框架支持 DPO 吗？

<critical_check>

ROLL 是 RLVR 框架，不一定原生支持 DPO。需要确认。

</critical_check>

让我列实际可用工具:

### Option 1: ROLL 内 SFT pipeline + 自定义 DPO loss
- 已有 sft_pipeline.py + sft_worker.py
- 需加 DPO loss 函数 (~50 行代码)
- 替换 CE loss

### Option 2: 用 TRL (Hugging Face) 独立做 DPO
- TRL 0.7+ 原生 DPOTrainer
- 不依赖 ROLL/Megatron
- 但需要单独 docker image 或 TRL 兼容 stage 8 HF ckpt

### Option 3: 简单粗暴 RFT + SFT (方案 A v1, 不做 DPO)
- 已有 ROLL SFT pipeline 直接可用
- 风险: 退化为 v1 方案 A 风险

**推荐**: Option 2 (TRL DPO), 因为:
- TRL DPOTrainer 经过广泛验证
- 不动 ROLL 框架，零风险
- HF 格式直接兼容 stage 8 输出
- DEV docker 已能跑 trl/transformers

---

## 5. 详细执行计划 (v2)

### Day 1: Rollout sampling + 分桶
- 配 vLLM serve stage 8 (TP=1, 1 GPU)
- 写 `scripts/data/gen_self_rollouts.py`
- 9259 题 × N=8 sample (T=0.7, max_tokens=512), `\boxed{X}` 提取
- 分桶: all-correct / partial / all-wrong
- **验证假设**: 实测 partial 题数是否 ~1800 (如完全不一致, 重新评估方案)

### Day 2 (Phase 2): DPO training
- 在 partial 题里 build pairs: 取 1 random correct + 1 random wrong
- 写 DPO training script (TRL DPOTrainer)
- 训练: 1 epoch, lr 5e-7, batch 16, beta 0.1 (DPO 标准)
- Save HF model

### Day 3: Stage 11a eval
- HF model → vLLM serve
- 全 5718 + clean 4551 + subset500 (direct + cot + del_tom)
- **Go/no-go decision**: 若 ≥ stage 8 → Phase 3, 否则 STOP

### Day 4 (Phase 3, 条件): GPT-5.5 short CoT 补 all-wrong
- 跑 `scripts/data/gen_short_cot.py` for ~280 all-wrong 题
- Audit 抽样 30 条 (~$3)
- SFT in TRL on 11a base, 280 records, 1 epoch, lr 5e-7

### Day 5: Stage 11b eval + 报告
- 同 Day 3 eval pipeline
- 写 `docs/stage11_report.md`
- 决定是否产 production v2.0

**总预算**:
- GPU: 1 GPU × 4h serving + 8 GPU × 4h training × 2 phases = ~70 GPU-h
- API: $14 (Phase 3 only)
- 时间: 5 天

---

## 6. 与 v1 (单纯 RAFT) 的关键差异

| 维度 | v1 方案 A | **v2 升级版** |
|---|---|---|
| 训练算法 | SFT (退化风险) | **DPO (保留对比信号)** |
| 训练数据 | all 9259 correct rollouts | **only ~1800 partial pairs + 280 hard SFT** |
| 数据浪费 | 78% 答对题被重训 | 跳过, 不浪费 |
| All-wrong 题 | 抛弃 | **GPT-5.5 短 CoT 补救** |
| 框架 | ROLL SFT | **TRL DPOTrainer (零修改 ROLL)** |
| Init | stage 8 | stage 8 (相同) |
| 预期 raw | +1-2pp | **+1-3pp** (DPO + GPT-5.5 一起) |
| 风险 | medium-high (spurious correct) | **medium** (DPO 抗 spurious correct) |

---

## 7. 终极 fallback (如果方案 v2 也失败)

如果 stage 11a (DPO) 不超过 stage 8:

| 因为... | 推断 | 行动 |
|---|---|---|
| Partial 题数 << 1800 | s8 已经过拟合训练集, RAFT 无空间 | STOP, accept stage 8 |
| Partial 题足够但 DPO 没涨 | 数据本身没新信息 | 尝试 ExploreToM 程序化数据 |
| DPO 涨但 +<1pp | 边际收益小, 不值得继续 | Accept stage 8 + DPO 11a |

---

## 8. 一些被排除的方法 (有证据)

### 排除 Thinking Machines OPD (reverse KL)
- 需要 teacher logits
- GPT-5.5 API 不提供
- deepseek API 也不提供 token-level distribution

### 排除 GAD (Black-Box GAN)
- 复杂度极高 (训练 discriminator + RL)
- ToMBench 这种 short MCQ 上**无先例**
- 训练不稳定

### 排除 Multi-teacher peer review
- 我们仅有 GPT-5.5 + deepseek-v4-pro 2 个 teacher
- "Peer" review 需要 ≥3 teacher
- 边际收益不清

### 排除 SFT cold start (再次)
- Stage 9 已 falsify (-1.51pp)
- 无理由再试

### 排除 long CoT 重训
- Stage 9 已 falsify
- ToMBench short-inference 性质

---

## 9. 总结: v2 的核心改进

**v1 错在哪**: 把 stage 8 当作"还能教"的状态，但实际 stage 8 已经过拟合训练集 (78% all-correct)，再 SFT 训练集没空间。

**v2 修正**:
1. **重新发现 stage 8 的真正可学习区**：1800 partial 题，不是 9259 全部
2. **用 DPO 代替 SFT**：保留对错对比信号
3. **GPT-5.5 仅修最难 (~280 all-wrong) 题**：不污染学生已对的
4. **TRL DPOTrainer**：零修改 ROLL，工具成熟

**预期最终**:
- Stage 11a (DPO only): 14B raw 0.76-0.78
- Stage 11b (DPO + GPT-5.5): **14B raw 0.78-0.80** (vs stage 8 0.7594)
- 距 deepseek 0.8080: 仍有 -1-3pp，但**接近反超**

如果 stage 11a 也不超 stage 8, 则**项目正式结束** — production_frozen/v1.0-production 是终态。

---

## 10. 参考文献 (新增)

### RAFT 风险 (避坑)
- [Beyond Rejection Sampling: Trajectory Fusion (2025)](https://arxiv.org/html/2602.04391) — spurious-correct trap
- [RIFT: Repurposing Negative Samples (2025)](https://arxiv.org/pdf/2601.09253) — false-positive 重利用

### DPO/IPO 视角 (升级)
- [It Takes Two: Your GRPO Is Secretly DPO (2025)](https://arxiv.org/html/2510.00977v3) — GRPO/DPO 等价性
- [Evaluating DPO and its Variants Across Tasks (ACL 2025)](https://aclanthology.org/2025.acl-srw.26.pdf)
- [A-IPO: Adaptive Intent-driven PO (2025)](https://arxiv.org/html/2510.10077v1)
- [On-Policy RL with Optimal Reward Baseline (2025)](https://arxiv.org/pdf/2505.23585)

### On-Policy Distillation (理论参考, 不可用因 logits)
- [Thinking Machines OPD blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Decoupling KL and Trajectories (2026)](https://arxiv.org/html/2605.16826)
- [Entropy-Aware On-Policy Distillation (2026)](https://arxiv.org/html/2603.07079v1)
- [KL for a KL: OPD with Control Variate (2026)](https://arxiv.org/html/2605.07865)

### Catastrophic Forgetting (缓解策略)
- [CSUR 2025 LLM Continual Learning Survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
- [Simple and Scalable Strategies to Continually Pre-train (OpenReview)](https://openreview.net/pdf?id=DimPeeCxKO)

### MCQ Distillation (直接对照)
- [LLM Distillation for Efficient Few-Shot MCQA (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.452.pdf) — **最直接相关的 paper**

---

最后更新: 2026-05-20 16:00
