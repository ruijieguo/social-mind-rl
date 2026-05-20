# 蒸馏方案深度调研 (Stage 11 候选)

> 基于 Stage 9 SFT-distillation 失败 (-1.51pp vs s8) 的反思 + 2025 年最新蒸馏方法调研。
> 目标：找到一个**真正适合 ToMBench 的蒸馏方案**，避开 Stage 9 已 falsify 的陷阱。

---

## 1. Stage 9 失败教训复盘（精确证据）

Stage 9 跑的是 **off-policy SFT distillation**: 用 GPT-5.5 生成 reasoning traces，让学生 SFT 模仿，再接 GRPO。结果：

| 失败现象 | 直接证据 |
|---|---|
| 模板冲突 | 5718/5718 outputs 含空 `<think></think>`，0% 含我们教的 `<reasoning>` |
| 长 CoT 害事 | wrong responses median 328 chars vs correct 234 chars |
| Intention task 大跌 | -6.3pp (vs s8) — short-inference 题被想多了 |
| GRPO KL 爆炸 | KL loss 0.034 → 7.258 at step 150（KL coef 0.001 不够） |

**根本教训**: GPT-5.5 写的是 **6-step 多步推理**，但 ToMBench 60% 是 **1-2 step shallow inference**。**Off-policy 蒸馏 teacher 的格式 + 内容**到学生时，学生用错地方了。

---

## 2. 蒸馏方法谱系 (2024-2025)

### 2.1 三大维度

| 维度 | 选项 |
|---|---|
| **数据来源** | (A) Off-policy: teacher 自己写 traces; (B) On-policy: student 写，teacher 评 |
| **监督信号** | (1) Logits (next-token distribution); (2) Sequence (text only); (3) Reward (binary correct/wrong) |
| **训练算法** | SFT / RFT (rejection sampling) / KL(student\|\|teacher) / RL with teacher reward |

### 2.2 具体方法对应表

| 方法 | 数据来源 | 监督信号 | 算法 | 教师要求 | 适合 ToMBench? |
|---|---|---|---|---|---|
| **Vanilla SFT distillation** (Stage 9) | Off-policy | Sequence | SFT (CE loss) | text only | ❌ FALSIFIED |
| **GKD (on-policy distillation)** | On-policy | Logits | KL on student tokens | **needs logits** | ❌ GPT-5.5 不提供 logits |
| **GAD (Black-Box On-Policy)** | On-policy | Text only | GAN-style discriminator | text only | ⚠ Complex setup |
| **RAFT / RFT (rejection sampling)** | On-policy student rollouts | Outcome reward (filtered) | SFT on correct only | **just verifier** | ✅ **Suitable** |
| **RAFT++** | On-policy | Outcome reward | SFT + low-variance bridge to GRPO | verifier | ✅ **Suitable** |
| **Multi-teacher OPD (AMPO)** | On-policy | Mixed | RL with off-policy teacher injection | logits+text | ⚠ Complex |
| **Expert Iteration / ReST** | On-policy | Outcome reward | SFT-iter | verifier | ✅ **Suitable** |
| **Reasoning trace SFT + RL (DeepSeek-R1 style)** | Off-policy → On-policy | Text + reward | SFT cold-start + RL | text | ❌ FALSIFIED on ToMBench (s9) |

### 2.3 关键 insight

[Agarwal et al. ICLR 2024 GKD](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf) 指出 **off-policy 蒸馏的核心缺陷**：student 在 inference 时 sample 自己的 tokens，但 training 时只见到 teacher 的 tokens — **分布漂移**。这正是 Stage 9 失败的根因（Qwen3 base 输出 `<think>`，但 SFT 教了 `<reasoning>`，分布漂移）。

**On-policy 蒸馏**让 student 自己写、teacher 评，分布对齐。

---

## 3. ToMBench 蒸馏的特殊约束 (有证据)

### 3.1 约束 1: GPT-5.5 不提供 logits

我们只能通过 OpenAI-compatible API 拿到 GPT-5.5 的 **answer + reasoning text**，**无 logits**。

排除：GKD (Agarwal 2024), DistilBERT-style logit distillation, KD with soft labels。
保留：Sequence-level distillation, RAFT/RFT, GAD-style 黑盒蒸馏。

### 3.2 约束 2: ToM 题是 short-inference 任务

Stage 9 retro 实测: ToMBench 60% 题在 stage 8 上已 ≥ 75% 答对（1-2 步推理足够），long CoT 反而引入错误。

排除：DeepSeek-R1 / Light-R1 风格的 long-CoT 蒸馏。
保留：**Answer-only 蒸馏** 或 **short CoT (≤3 步) 蒸馏**。

### 3.3 约束 3: 训练数据有 9259 已经高质量

Stage 8 的训练数据已经包括 GPT-5.5 合成 (Phase B/C, 2000 records)。再蒸馏需要 **新的数据视角**，而不是同样数据再用 GPT-5.5 写一遍。

### 3.4 约束 4: 评测协议是 direct (single decode, T=0)

排除 self-consistency / N-sample voting 的 test-time tricks。

**蒸馏目标**: 提升 student 在 **T=0 single decode** 下的能力。

### 3.5 约束 5: 验证答案需要 gold label

ToMBench 训练集每题有 gold answer。验证 student rollout 是否正确**不需要 teacher** — 只需 string match `\boxed{X}` vs gold。

**RAFT 不需要 GPT-5.5 调用** 来 filter rollouts。这极大降低成本。

---

## 4. 三个候选方案 (按 ROI 排序)

### 方案 A: ReST/RAFT++ - student self-distillation with gold-filter ⭐⭐⭐⭐⭐

**思路**:
1. 用 stage 8 14B (production-best) 在 9259 训练题上 sample N=8 答案 (T=0.7)
2. **Filter**: 只保留 student 自己答对的 rollouts (matches gold)
3. 把 filtered correct rollouts 当 SFT 数据
4. 继续 SFT 学生 1-2 epochs
5. 评测

**为什么这是最佳路径**:
- ✅ **完全 on-policy** (student 自己写，避免 stage 9 分布漂移)
- ✅ **不依赖 GPT-5.5** (gold label 是 verifier)
- ✅ **零成本** (除了 vLLM 推理 9259×8 = 73k samples, ~1-2h)
- ✅ **保留学生自己的推理风格** (不教 `<reasoning>` 标签，不教 long CoT)
- ✅ **已被多次证明** (RAFT, RAFT++)

**预期增益** (基于 RAFT 论文 + 我们的 stage 8 起点):
- Stage 8 correct rate on training set: ~85% (高，因为 stage 8 已训过该数据)
- 9259 × 8 = 73k samples, ~62k filtered correct
- SFT 1 epoch on 62k self-generated correct rollouts
- 预期 raw: 0.7594 → **0.77-0.78** (+1-2pp, RAFT 论文典型增益)

**风险**:
- 学生可能 "lock-in" 自己的错误推理风格 (correct-by-luck 问题)
- 缓解: filter step 不只看 final answer, 也 reject 异常短 (<20 tokens) 和异常长 (>800 chars) 的 rollouts

参考: [A Minimalist Approach to LLM Reasoning, arxiv 2504.11343](https://arxiv.org/html/2504.11343v1), [RLHFlow/RAFT](https://github.com/RLHFlow/RAFT)

---

### 方案 B: GPT-5.5 short-answer guided RFT ⭐⭐⭐⭐

**思路**:
1. 用 student (stage 8) sample N=4 答案 / 题 (T=0.7)
2. **如果学生没答对**: 用 GPT-5.5 写一个**简短** (1-3 step, ≤200 chars) 推理 + 答案
3. **如果学生答对**: 保留学生 rollout
4. 混合数据 SFT 1 epoch

**为什么有道理**:
- 学生答对的题 → 保留 student style (避免分布漂移)
- 学生答错的题 → GPT-5.5 给出 **短** 推理示范 (避免 stage 9 long-CoT 问题)
- 不教 `<reasoning>` 标签，所有推理直接 inline，不强制结构

**关键改进 vs Stage 9**:
- ✅ **短推理强制** (≤200 chars, ≤3 steps) — 解决 Stage 9 long-CoT 问题
- ✅ **只教错的题** — 不污染学生答对的题
- ✅ **不教标签** — 解决 Stage 9 `<reasoning>` vs `<think>` 冲突

**预期增益**:
- Stage 8 wrong rate on training: ~15% (因为 stage 8 训过)
- 9259 × 0.15 = ~1400 题需要 GPT-5.5 short reasoning
- 1400 × $0.04 ≈ $56 cost
- 预期 raw: 0.7594 → **0.78-0.80** (+2-4pp，如果 GPT-5.5 的短推理真能帮助 ToM 答题)

**风险**:
- 短推理可能仍有 stage 9 的标签/分布问题 (但风险小，因为我们用 inline 不用 tag)
- GPT-5.5 短推理在 1400 题上质量需要 audit

参考: [SHAD+RFT token-level weighting](https://www.emergentmind.com/topics/rejection-fine-tuning-rft)

---

### 方案 C: Black-Box On-Policy GAD ⭐⭐

**思路** ([arxiv 2511.10643](https://arxiv.org/abs/2511.10643)):
1. 训练一个判别器: 区分 student vs teacher 输出
2. Student 通过 RL 学着骗过判别器
3. Student 不需要看到 teacher logits

**为什么暂不推荐**:
- ❌ 实施复杂 (GAN-style, 训练判别器 + RL)
- ❌ 在 ToMBench 这种短题型上**没有先例**
- ❌ 训练不稳定，调参困难
- ⚠ 论文有效果但目标场景是 general chat 不是 MCQ

预期: 不确定，可能 +0-3pp，开发成本 7-10 天。

---

### 方案 D: Multi-teacher Peer Review ⭐⭐⭐

**思路**: [Reasoning Distillation from Mixture of Teachers with Peer Review (ACL 2025)](https://aclanthology.org/2025.findings-acl.217.pdf)
- 用多个 teacher (GPT-5.5 + deepseek-v4-pro + ...) 各写 reasoning
- Peer review: 让 teacher 们互相评分
- 选 consensus 高的作为 SFT 数据

**问题**: 我们目前只有 GPT-5.5 + deepseek-v4-pro API。复杂度大，相对方案 A/B 边际收益不清晰。

---

## 5. 推荐：方案 A → 方案 B (ablation 顺序)

### 第一步: 跑方案 A (zero-cost, 4-6 hours)
**Why first**: 完全不依赖 GPT-5.5，纯 self-distillation。如果 work，我们就有 +1-2pp 且方法学上"干净"（无外部依赖）。

```
Day 1: 实现 self-rollout + filter pipeline (~2h)
Day 1: vLLM serve stage8 + 9259 * 8 samples (~2-3h)
Day 2: SFT 1 epoch on filtered ~62k rollouts (~3h)
Day 2: Eval, 对比 stage 8
```

### 第二步: 如果方案 A 成功 → 跑方案 B 叠加 (1-2 days)
**Why second**: 方案 A 没解决"学生原本就错的题"。方案 B 用 GPT-5.5 短推理补这部分。

```
Day 3: 用 stage-after-A serve + sample 4 / 题 → 找 wrong
Day 3: GPT-5.5 生成 short reasoning for wrong (~1400 题, $56)
Day 4: Audit short reasoning quality (~10% sample, $20)
Day 4: SFT on combined (A correct + B short-teacher) data
Day 5: Eval
```

### 如果方案 A 失败
**Decision**: 不再继续蒸馏路线，确认 stage 8 是 GRPO+data 上限。考虑 32B 规模或 ExploreToM 程序化数据。

---

## 6. 关键差异 vs Stage 9 (避坑列表)

| Stage 9 错误做法 | 新方案改正 |
|---|---|
| Off-policy SFT (teacher writes, student mimics) | **On-policy** (student writes, filter or teacher fixes only wrong) |
| 教 `<reasoning>` 标签 | **不教任何新标签** (inline reasoning, 学生自然风格) |
| GPT-5.5 4-6 步长 CoT | **强制短 reasoning** (≤200 chars, ≤3 steps if used) |
| 训练所有题 (包括学生已会的) | **只训学生不会的 (B) 或学生正确版本 (A)** |
| SFT → RL (lr 5e-7, KL=0.001) | **SFT only, 1 epoch, no RL on top** |
| `<think></think>` 模板冲突未察觉 | **训前手动 prompt-format 验证**: rollout 100 题看输出是否符合期望 |

---

## 7. 具体配置 (方案 A 详细)

### 7.1 Data generation

```python
# scripts/data/gen_self_rollouts.py
# For each of 9259 training records:
#   sample N=8 from stage 8 model (T=0.7, max_tokens=512)
#   extract \boxed{X} from each
#   keep ones matching gold + length 50-500 chars

# Expected output: ~60-70k correct rollouts
```

### 7.2 SFT data format

直接用 student 自己的 rollout（包括它自然产生的 `<think></think>` 包装），不改造：

```json
{
  "messages": [
    {"role": "system", "content": "<original system prompt>"},
    {"role": "user", "content": "Story:\n...\nQuestion: ...\nA. ...\nB. ...\nC. ...\nD. ..."},
    {"role": "assistant", "content": "<student's own rollout text, including \\boxed{X}>"}
  ]
}
```

### 7.3 SFT config

```yaml
# sft_config_stage11_self_distill_14b.yaml
pretrain: /mnt/output/qwen3-14B-tom-hf-stage8  # init from stage 8 (not base!)
data: /mnt/data/tom_train_self_distill.jsonl   # ~62k filtered rollouts
sequence_length: 1024
num_train_epochs: 1                            # 1 epoch only
learning_rate: 1.0e-6                          # 极低 lr (在 stage 8 之上微调)
warmup_steps: 30
save_steps: 200
save_hf_model: true
```

### 7.4 关键不变 (从 stage 8 继承)

- **No KL loss**
- **No long CoT prompt**
- **No `<reasoning>` tag teaching**
- **No SFT before stage 8** (we init FROM stage 8, not from base)
- **response_length 不动** (just inherit stage 8's 256 if any post-RL)

---

## 8. 预算

| 方案 | API 成本 | GPU-hours | 实施天数 | 预期增益 |
|---|---|---|---|---|
| A (Self-distill) | $0 | ~6 (1 GPU vLLM 1h + 8 GPU SFT 3h) | 2 | +1-2pp |
| B (A + short teacher) | $80 ($56 gen + $24 audit) | ~10 | +3 | +2-4pp |
| C (GAD) | ~$200 | ~30 | 7-10 | 不确定 |
| D (Multi-teacher) | ~$300 | ~12 | 5 | +1-3pp |

**首选 A → B**: 总 5 天 + ~$80 + 16 GPU-h，预期 **14B raw 0.78-0.80**。

---

## 9. 风险与缓解

### 风险 1: Stage 8 的 correct rollouts 反映已知问题
**证据**: Stage 8 错 1376 题（24% wrong）。其中 40% 是标签噪声，60% 真错。Self-distill 只能保留 already-correct 的，**对那 24% wrong 无作用**。
**缓解**: 这正是方案 B 的目的 — 用 GPT-5.5 短推理修学生 wrong 的题。

### 风险 2: vLLM determinism
**证据**: vLLM 不同 batch 大小 / TP 配置下 T=0.7 输出可能有微小差异。
**缓解**: 用 T=0 在 stage 8 上跑一次作为 sanity check (期望命中率 ~85%)，与 T=0.7 N=8 的平均命中率对比。

### 风险 3: Filter 太严，data 量不够
**证据**: 假设最差 stage 8 在 train set 上 75% 命中率，N=8 → 6 correct/题 × 9259 = 55k。仍够。
**缓解**: 如不够，加 sample N=16。

### 风险 4: SFT 收敛后 model 变 narrow（只能答原训练数据风格）
**证据**: SFT 本身倾向 overfit。
**缓解**: 严格 1 epoch + lr 1e-6 + 监控 val_correct 上 hold-out subset500。

---

## 10. 决策矩阵

| 选项 | 数据成本 | 计算成本 | 时间 | 预期效果 | 风险 | 选 |
|---|---|---|---|---|---|---|
| **A: Self-distill** | $0 | 6 GPU-h | 2 days | +1-2pp | low | ⭐ **首选** |
| B: A + GPT-5.5 fix wrong (short CoT) | $80 | 10 GPU-h | 3 days | +2-4pp | medium | 第二步 |
| C: GAD black-box | $200 | 30 GPU-h | 7 days | 不确定 | high | ❌ |
| D: Multi-teacher peer | $300 | 12 GPU-h | 5 days | +1-3pp | medium | ❌ |

---

## 11. 总结

**Stage 9 失败的根因**: off-policy distillation + long CoT + tag injection 三连击，与 ToMBench short-inference 形态冲突。

**Stage 11 (新方案 A): student self-distillation via gold-filter RAFT**:
- 完全 on-policy → 无分布漂移
- 不教任何新格式 → 无 tag 冲突
- 无外部 teacher → 无 long-CoT 注入
- Init from stage 8 → 继承已有的所有改进
- 评测协议不变 → direct T=0 single decode

**预期最终上限** (有 stage 1-10 + 文献证据基础):
- Stage 8 baseline: 14B raw 0.7594, clean 0.8449
- After Stage 11 方案 A: **14B raw 0.77-0.78, clean 0.86-0.87**
- After Stage 11 方案 A+B: **14B raw 0.78-0.80, clean 0.87-0.89**
- 距 deepseek-v4-pro 0.8080: -0-3pp（**可能反超 raw**！）
- 距 GPT-5.5 0.8349: -3.5-5.5pp（**clean eval 上可能接近**）

---

## 参考文献

### 蒸馏基础
- [On-Policy Distillation of Language Models (GKD, ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)
- [A Survey of On-Policy Distillation for Large Language Models (2025)](https://arxiv.org/pdf/2604.00626)

### 黑盒蒸馏
- [Black-Box On-Policy Distillation of LLMs (GAD, 2025)](https://arxiv.org/abs/2511.10643)
- [Reasoning Distillation from a Mixture of Teachers with Peer Review (ACL 2025)](https://aclanthology.org/2025.findings-acl.217.pdf)

### RAFT / Rejection Sampling
- [A Minimalist Approach to LLM Reasoning: Rejection Sampling to Reinforce (2025)](https://arxiv.org/html/2504.11343v1)
- [RLHFlow/RAFT](https://github.com/RLHFlow/RAFT)
- [RLHFlow/Minimal-RL](https://github.com/RLHFlow/Minimal-RL)
- [Rejection-sampling Fine-Tuning (RFT) overview](https://www.emergentmind.com/topics/rejection-sampling-fine-tuning-rft-ad4c417c-416b-40b6-bf9a-4653b83ddcfb)
- [RLHF Book - Rejection Sampling chapter (Nathan Lambert)](https://rlhfbook.com/c/09-rejection-sampling)

### 我们已 falsify 的方法
- Stage 9 retrospective: `docs/stage9_retro.md` - SFT cold start failed on ToMBench
- Stage 10 retrospective: `docs/stage10_plan_evidence_based.md` - weighted_sum reward failed

### 自反思
- [SRT: Can Large Reasoning Models Self-Train?](https://self-rewarding-llm-training.github.io/) - self-consistency 路线 (排除，违反 single-decode 约束)

---

最后更新: 2026-05-20 14:00
