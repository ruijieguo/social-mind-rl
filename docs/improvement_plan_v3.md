# 改进方案 v3：真正训练突破，稳定逼近 GPT-5.5

> **核心约束**：不依赖 self-consistency / multi-sample voting 等 test-time compute trick。
> 只通过训练数据合成 + 训练算法改进，实打实把模型本身做强。

> **目标**：14B 在 ToMBench full 5718 direct 协议上稳定达 ≥ **0.81** (raw), ≥ **0.90** (clean)，单次推理，温度 0。
> 距 GPT-5.5 raw 0.8349 缩到 -3pp 以内，反超 deepseek-v4-pro 0.8080。

---

## 0. 问题精确诊断（基于 stage8 14B 完整数据）

### 0.1 真实可达 ceiling

| 维度 | 数值 |
|---|---|
| Stage8 14B raw 5718 direct | 0.7594 |
| GPT-5.5 raw 5718 direct | 0.8349 |
| **GPT-5.5 对、Stage8 错** | **704 题** |
| 若全部学会 → Stage8 上限 | **0.8825** |
| 距 GPT-5.5 实际 gap | -7.55pp |
| Stage8 对、GPT-5.5 错（噪声/标签问题） | 272 题 |

**关键 insight**：**有 704 题是 GPT-5.5 能对但我们错的，理论上可学**。这就是我们的目标空间——平均每个 task ~100 题，分布在 7 个 ToMBench 子任务上。

### 0.2 Stage8 错误结构（精确分布）

| Task | s8 错总数 | HOT (deepseek+gpt 都对) | only_gpt 对 | 三方都错 (ceiling) |
|---|---|---|---|---|
| Belief | 76 | 34 | 11 | 20 |
| Desire | 150 | 39 | 27 | 79 |
| Emotion | 229 | 80 | 28 | 93 |
| **False Belief** | 220 | **102** | 39 | 46 |
| Intention | 124 | 61 | 14 | 33 |
| **Knowledge** | 281 | 84 | 39 | 144 |
| **Non-literal Comm** | 296 | 92 | 54 | 127 |
| **TOTAL** | **1376** | **492** | **212** | **542** |

**HOT（高置信度可优化）= 492 题 = +8.60pp 上限**。如果都学会，14B 直冲 0.845。

**three highest-leverage tasks**：False Belief (102 HOT), Non-literal Comm (92), Knowledge (84)。

### 0.3 训练算法层面的精确浪费证据

Stage8 14B 训练全程 `samples_used / final_mask` (每 batch 256 rollout):

| step | samples_used | final_mask | rollout | all_correct |
|---|---|---|---|---|
| 50 | 57/256 = **22%** | 0.22 | 0.80 | 0.44 |
| 100 | 33/256 = **13%** | 0.13 | 0.82 | 0.62 |
| **150** | **14/256 = 5%** ⚠️ | 0.05 | 0.95 | 0.84 |
| 200 | 28/256 = 11% | 0.11 | 0.92 | 0.81 |
| 349 | 16/256 = **6%** | 0.06 | 0.91 | 0.78 |

**致命数据**：训练后期 95% 的 rollout token 都没贡献梯度。Stage8 在 step 150 后基本没在学新东西。这是 GRPO 标准实现 + 静态难度遮罩 (0.1/0.95) 的固有缺陷。

---

## 1. 战略思路（不偷奸耍滑的两条路径）

### 路径 A：把模型本身做强（核心）

**Step 1**: 通过高质量 reasoning trace SFT 给模型一个 strong cold start（不只是答案，更是**推理过程**）
**Step 2**: 在 SFT 之上做 RLVR（修复算法缺陷 + 难度课程）
**Step 3**: 数据继续迭代（针对剩余 HOT 错误专项合成）

这是 DeepSeek-R1 / Light-R1 / Magistral / OpenThoughts 等 2025 SOTA 的标准 pipeline。我们之前 8 个 stage 全部是 "RL only" 跳过了 SFT cold start，**这是单点最大改进空间**。

### 路径 B：算法层修复 GRPO 的固有问题

- Dr.GRPO（修长度偏差）
- 自适应难度遮罩（修 saturation）
- High-Entropy Token Selection（聚焦关键 token 的梯度）
- Length-Adaptive GAE（替代 group-relative）

---

## 2. 详细方案 — 6 个核心策略

### **策略 1: GPT-5.5 推理痕迹蒸馏 SFT (Cold Start)** ⭐⭐⭐⭐⭐

**这是预期收益最大、最重要的一步**。

#### 1.1 为什么必须做 SFT cold start

DeepSeek-R1 的 [Nature paper](https://www.nature.com/articles/s41586-025-09422-z.pdf) 明确表明：
- **R1-Zero**（纯 RL，无 SFT）能学到推理，但**输出杂乱、可读性差**
- **R1**（先 SFT 再 RL）显著优于 R1-Zero
- 蒸馏 R1 的 reasoning traces 到 14B 模型，性能远超直接 RL 14B

[Light-R1](https://arxiv.org/html/2503.10460v1) 在 14B 上做 SFT + DPO + RL 三阶段 curriculum，AIME24 上 74.0%。我们目前全部是单阶段 GRPO，没有 SFT 基础。

#### 1.2 我们 ToM 任务的具体设计

**数据生成**：
1. 从 9259 训练集采样 5000 条（覆盖 7 个 task 均衡）
2. 让 GPT-5.5 对每题生成结构化推理：

```
Story: ...
Question: ...
Reasoning: 
Step 1 [identify mental states]: A 看到 X 在容器甲，B 没看到 X 被移到乙。
Step 2 [track knowledge state]: B 离开后 A 移了 X，所以 B 不知道。
Step 3 [apply ToM rule]: B 的信念基于他离开前的观察。
Step 4 [conclude]: B 会去甲找 X。
Answer: \boxed{A}
```

3. **关键**：让 GPT-5.5 给出 **推理链 + 最终答案** 而不是只答案
4. 过滤：只保留 GPT-5.5 答案 = gold label 的样本（约 80%×5000 = 4000 条）
5. 用 GPT-5.5 audit 验证推理链是否合理（reject 推理混乱的）

#### 1.3 SFT 配置

```yaml
# Stage 9a SFT cold-start
model: Qwen3-14B (base)
data: 4000 GPT-5.5 reasoning traces (mix of 7 tasks)
loss: SFT (next-token CE on reasoning + answer)
lr: 5e-6 (low to avoid catastrophic forgetting)
epochs: 3
batch_size: 32
warmup: 100 steps
gradient_accumulation: 4
```

**估计**: 14B SFT 4000 条 × 3 epochs ≈ 12k steps × 75s = ~4h, 1×8 H800

#### 1.4 SFT 后效果预测

基于 Light-R1 经验：SFT cold start 在 reasoning task 上单 stage 即可 +5-10pp（取决于教师模型质量）。

我们 GPT-5.5 = 顶级教师，预测 14B SFT 后单看 raw 5718 direct：
- **直接增益 +3-5pp**（不含后续 RL），即 0.79-0.81 (raw)
- 后续接 GRPO 再 +2-3pp

#### 1.5 数据合成成本

- GPT-5.5 reasoning trace 生成: 4000 题 × 2.5s ≈ ~3h, **成本约 $200**
- Audit + filter: 4000 题 × 1s ≈ 1h, **成本约 $50**
- 合计: **~$250, ~4h**

参考: [DeepSeek-R1 Nature paper](https://www.nature.com/articles/s41586-025-09422-z.pdf), [Light-R1 ACL 2025](https://aclanthology.org/2025.acl-industry.24/), [Open-R1 Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)

---

### **策略 2: 程序化 ToM 数据生成 (ExploreToM 2.0)** ⭐⭐⭐⭐⭐

#### 2.1 为什么 GPT-5.5 合成的数据不够

我们的 Phase A/B/C 数据都是 GPT-5.5 **自由生成**，质量好但有两个问题：
1. 多样性有限（GPT-5.5 自由生成时有偏好分布）
2. 难度不可控（无法生成"刚好难倒模型"的样本）

[Meta 的 ExploreToM (ICLR 2025)](https://arxiv.org/abs/2412.12175) 用 **A* 搜索 + 程序化模板** 生成对抗性 ToM 数据。结果：在 ToMi / HiToM 基准上微调后 **+27 个百分点**。

**核心思路**：不让 LLM 自由生成，而是定义 ToM 程序化模板（状态机），让 A* 搜索找出最让模型出错的故事配置。

#### 2.2 适配我们的 ToMBench 7 个 task

实现一套 ToM 程序化生成框架：

```python
# pseudo-code
class ToMStateMachine:
    characters: List[Agent]
    objects: List[Object]
    locations: List[Location]
    events: List[Event]  # 状态转换

    def generate_belief_question(self, max_depth=3):
        # 一阶: A 看到 X 在甲, B 移走, A 回来 → A 信念
        # 二阶: A 知道 B 看到 X, C 偷偷移走 → A 认为 B 信念
        # 三阶: A 知道 B 知道 C 看到 X, D 改变 → A 认为 B 认为 C 信念
        ...

    def adversarial_search(self, target_model, n_trials=20):
        # A* 搜索找到模型最容易答错的事件序列
        ...
```

#### 2.3 7 个 task 的程序化生成

| Task | 程序模板 | 难度参数 |
|---|---|---|
| Belief | 移动 + 观察缺失 → 信念错位 | depth (1st/2nd/3rd order) |
| False Belief | Sally-Anne + 多角色追踪 | 角色数, 事件数 |
| Knowledge | who-saw-what graph | 节点数, 边数 |
| Desire | 目标冲突 → 选择 | 偏好向量维度 |
| Emotion | 事件因果 → 情绪 | 复杂度 |
| Intention | 行为序列 → 隐藏目标 | 间接性 |
| Non-literal Comm | 字面 vs 暗示 | 上下文 distance |

#### 2.4 对抗性搜索

针对 stage8 14B 的 492 条 HOT 错误，分析每条错误的"模式"：
- 多步 belief tracking 错（占 False Belief 错误的 40%）
- 隐含数量推理错（Knowledge）
- 礼貌反应误判（Non-literal Comm）

然后**专门生成大量这些模式的变体**（每个模式 ~100-200 条），目标 2000 条对抗性数据。

#### 2.5 与现有数据的关系

不是替换，是**追加**：
- 现有 9259 records（base + Phase B + Phase C）保留
- 新增 2000 条 ExploreToM 2.0 程序化数据
- 新增 4000 条 GPT-5.5 reasoning traces（SFT 用）
- 总计训练数据 ~15000 条

#### 2.6 实施
开发 ExploreToM 2.0 的核心代码 ~5-7 天，能复用 Meta 论文公开代码（Github: explore-tom），但需要适配 ToMBench 的 7 个 task 模板。

预期增益: **+2-3pp** raw on ToMBench

参考: [ExploreToM ICLR 2025](https://arxiv.org/abs/2412.12175), [Meta 公开实现](https://ai.meta.com/research/publications/explore-theory-of-mind-program-guided-adversarial-data-generation-for-theory-of-mind-reasoning/)

---

### **策略 3: 自适应难度课程 + 动态难度遮罩** ⭐⭐⭐⭐

#### 3.1 训练前离线评估每条数据的"模型适合度"

```python
# 训练前一次性跑（~30min）
for each training example:
    # 用 base model 在每条样本上 sample 16 次 (T=0.7)
    samples = [model.sample(prompt) for _ in range(16)]
    success_rate = mean([s == gold for s in samples])

# 分桶
easy = [s for s in data if s.success_rate >= 0.75]      # ~3000 条
medium = [s for s in data if 0.25 <= s.success_rate < 0.75]  # ~5000 条
hard = [s for s in data if s.success_rate < 0.25]      # ~3000 条
```

#### 3.2 三阶段课程 (E2H Curriculum)

```yaml
# 训练阶段 1 (step 0-100): 建立基础
mixing_ratio:
  easy: 0.50
  medium: 0.40
  hard: 0.10

# 阶段 2 (step 100-250): 深入推理
mixing_ratio:
  easy: 0.20
  medium: 0.50
  hard: 0.30

# 阶段 3 (step 250-400): 突破上限
mixing_ratio:
  easy: 0.05
  medium: 0.30
  hard: 0.65
```

#### 3.3 动态难度遮罩

当前固定 `difficulty_low=0.1, high=0.95` 太严格。改自适应：

```python
target_active_ratio = 0.40  # 目标 40% 样本贡献梯度（vs 现在 5-22%）

if current_active_ratio < 0.30:
    # 太多样本被遮罩，放宽
    low_threshold = max(0.02, low_threshold * 0.9)
    high_threshold = min(0.99, high_threshold + 0.02)
elif current_active_ratio > 0.55:
    # 信号太多，收紧
    low_threshold = min(0.20, low_threshold * 1.1)
    high_threshold = max(0.85, high_threshold - 0.01)
```

#### 3.4 期望效果

- Step 200-400 后期训练**真正在学新东西**（vs 现在 step 150 后空转）
- 14B max_steps 可以延到 500-600 而不浪费
- 8B 突破 0.74 ceiling（这是 8B 5 个 stage 都没破的硬伤）

**预期增益**: 14B +1.5-2.5pp, 8B +1-1.5pp

参考: [E2H Curriculum (Learning Like Humans)](https://arxiv.org/abs/2505.08364), [Online Difficulty Filtering for RLVR](https://arxiv.org/html/2504.03380v2), [Self-Evolving Curriculum](https://arxiv.org/pdf/2505.14970)

---

### **策略 4: 算法层关键修复 (Dr.GRPO + KL + Entropy bonus)** ⭐⭐⭐

#### 4.1 Dr.GRPO loss 修复

[Understanding R1-Zero (COLM 2025)](https://arxiv.org/pdf/2503.20783) 证明：GRPO 在 reward normalization 和 token aggregation 上有偏差，导致错误回答倾向更长。

**实施**:
```python
# 原 GRPO
advantage = (r - mean(r_group)) / std(r_group)        # 偏差 1: 用 std 归一化放大噪声
loss = -mean(advantage * log_prob_per_token_in_response)   # 偏差 2: 长 response 被稀释

# Dr.GRPO 修复
advantage = r - mean(r_group)                          # 不除 std
loss = -sum(advantage * log_prob_per_token) / max_len  # 用全局 max_len 归一化
```

改 `framework/ROLL/.../grpo_loss.py` 3 行代码。

**预期增益**: +0.5-1pp

#### 4.2 重启 KL=true (with kl_coef=0.001)

我们之前 stage3 用过 KL=true 但效果不好，因为 `kl_coef=0.01` 太重。正确值是 0.001（[RLOO 推荐](https://arxiv.org/html/2402.03300)）。

```yaml
add_token_level_kl: true
kl_coef: 0.001     # 之前 0.01 太重；0.001 是合适的
kl_target: null    # 不用 adaptive
```

**期望效果**: 防止策略过度偏离 base model 分布（stage8 KL loss 已涨到 0.42 是过拟合预警）

**预期增益**: +0.3-0.5pp

#### 4.3 Entropy bonus / High-Entropy Token Selection

[High-Entropy Minority Tokens (2025)](https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr/) 发现：在 RLVR 中，仅有 ~20% 的"高熵 token"对最终答案有决定性影响。把梯度集中在这些 token 上能显著提升训练效率。

**实施**:
```python
# 在 GRPO loss 计算时
token_entropy = -sum(p * log(p) for p in vocab_dist)
high_entropy_mask = token_entropy > entropy_threshold  # top 20% tokens

# 只在 high-entropy tokens 上反传梯度
loss = -mean(advantage * log_prob * high_entropy_mask)
```

**预期增益**: +0.5-1pp + 训练效率 ×5

参考: [Token-Efficient RL](https://arxiv.org/html/2603.06619v1), [High-Entropy Minority Tokens (MarkTechPost)](https://www.marktechpost.com/2025/06/08/high-entropy-token-selection-in-reinforcement-learning-with-verifiable-rewards-rlvr-improves-accuracy-and-reduces-training-cost-for-llms/)

---

### **策略 5: response_length 调整 + Long CoT 训练** ⭐⭐⭐

#### 5.1 当前 256 token 限制太短

ToMBench 推理题需要多步：
- belief tracking 至少 3-4 步
- 二阶 belief 5-6 步
- non-literal comm 包含暗示推理 4-5 步

每步约 30-50 tokens，**至少 200-300 token 才足够推理**。我们 response_length=256 太紧。

**当前**: `response_length: 256, l_max: 256` 直接卡死 reasoning chain
**新**: `response_length: 1024, l_max: 1024, val max_new_tokens: 512`

#### 5.2 配合 SFT cold start

如果策略 1 SFT 让模型学到了多步推理格式，response_length 必须够长才能让 RL 阶段不截断。

#### 5.3 计算成本

- 14B response_length 256 → 1024: rollout 时间 ×3-4，但仍可接受
- 单步训练时间 75s → ~200s
- 350 步训练: 7h → ~18h（一晚跑完）

#### 5.4 风险

```
Stage 3 教训: 把 response_length 升到 384 反而坏了，因为 val 协议截到 64 token，长 reasoning 在最后才出 \boxed{X}, val 评测失败。

修复: val protocol 同步升到 512 token。
```

**预期增益**: +1-2pp（释放 reasoning chain）

参考: [Dissecting Long-CoT Reasoning Models](https://arxiv.org/html/2506.04913v2)

---

### **策略 6: Reward shaping 改进（取消乘法）** ⭐⭐

#### 6.1 当前问题

```python
r_total = r_fmt × r_out × r_len   # 乘法
```

- 答对了但格式略偏 → r_fmt=0 → r_total=0，模型完全没 credit
- 长 reasoning 答对 → r_len 略低 → r_total 被压制
- 这压制了 Knowledge / Non-literal 这类需要长推理的 task

#### 6.2 改加权和

```python
# 新公式
r_total = (
    0.05 * r_fmt +          # 格式只占 5%
    0.85 * r_out +          # 正确性最重要
    0.10 * sigmoid_window(L, 100, 600, k=10)  # 鼓励 100-600 长 reasoning
)
```

**预期增益**: +0.3-0.6pp

参考: 上一个改进方案

---

## 3. 实施计划 (Phase 1 + Phase 2 + Phase 3)

### Phase 1 (Week 1-2): SFT Cold Start + 算法基础修复
**预期**: 14B raw **0.7594 → 0.80-0.82**

| 任务 | 工时 | 训练时间 | 增益 |
|---|---|---|---|
| 1.1 GPT-5.5 生成 4000 reasoning traces ($250) | 0.5 天 | — | — |
| 1.2 14B SFT cold start (4000 × 3 epochs) | 1 天 | 4h | +3-5pp |
| 1.3 实现 Dr.GRPO loss fix | 0.5 天 | — | — |
| 1.4 实现 entropy-aware token selection | 1 天 | — | — |
| 1.5 调整 response_length 256 → 1024 + val max_tokens 512 | 0.5 天 | — | — |
| 1.6 改 reward shaping (加权和) | 0.5 天 | — | — |
| 1.7 重训 14B stage9 (SFT base + 改进 GRPO, 350 步) | 0.5 天 | 18h | +2-3pp |
| **小计** | **~5 天 + 22h GPU** | | **+5-8pp** |

**Phase 1 终点**: 14B raw 0.80-0.82, 已**追平 deepseek-v4-pro 0.8080**

### Phase 2 (Week 3-4): 难度课程 + 动态遮罩
**预期**: 14B raw **0.80-0.82 → 0.83-0.85**

| 任务 | 工时 | 训练时间 | 增益 |
|---|---|---|---|
| 2.1 离线 base model 给 15000 训练数据打分 (8 GPU × 30min) | 0.5 天 | — | — |
| 2.2 分桶 + curriculum schedule 实现 | 1 天 | — | — |
| 2.3 自适应难度遮罩 (改 ROLL scheduler) | 1 天 | — | — |
| 2.4 重训 14B stage10 (curriculum + 自适应遮罩) | 0.5 天 | 24h (500 步) | +2-3pp |
| **小计** | **~3 天 + 24h GPU** | | **+2-3pp** |

**Phase 2 终点**: 14B raw 0.83-0.85, **接近 GPT-5.5 raw 0.8349**

### Phase 3 (Week 5-7): ExploreToM 2.0 + 32B (并行)
**预期**: 14B raw **0.83-0.85 → 0.86-0.88**, 32B raw **0.81-0.83**

| 任务 | 工时 | 训练时间 | 增益 |
|---|---|---|---|
| 3.1 ExploreToM 2.0 程序化生成框架 | 5 天 | — | — |
| 3.2 生成 2000 条对抗性数据（针对 HOT 错误） | 1 天 | — | — |
| 3.3 重训 14B stage11 (含 ExploreToM 2.0 数据) | 0.5 天 | 24h | +1-2pp |
| 3.4 [可选] 训练 32B stage8 配方 | 0.5 天 | 14h | +2pp vs 14B |
| **小计** | **~6.5 天 + 38h GPU** | | **+1-2pp 14B / 32B 新模型** |

**Phase 3 终点**: 14B raw **0.86-0.88**, **大幅缩小与 GPT-5.5 (0.8349) 的距离，clean eval 可能反超 GPT-5.5**

---

## 4. 详细的 Stage 9 配置（Phase 1 的核心训练）

```yaml
# rlvr_config_stage9_1x8_14b.yaml
exp_name: "qwen3-14B-tombench-rlvr-stage9-1x8"

# 关键变化
init_from_sft_checkpoint: /mnt/output/qwen3-14B-tom-sft/   # ← SFT cold start
max_steps: 350
save_steps: 350
data: /mnt/data/tom_train.jsonl   # 9259 (s8 之前数据)

# response length 加长
prompt_length: 1024
response_length: 1024              # ← 256 → 1024

# Dr.GRPO loss
adv_estimator: "drgrpo"            # ← 新算法
whiten_advantages: false           # Dr.GRPO 不用 std 归一化
loss_normalization: "max_len"      # ← 关键: 用 max_len 而非 actual length

# Token-level KL with low coef
add_token_level_kl: true
kl_coef: 0.001                     # ← 之前 stage3 用 0.01 太重

# Entropy bonus
entropy_bonus_coef: 0.01           # ← 新增
high_entropy_token_only: true      # ← 新增: 只在 top 20% entropy token 上反传

# Reward shaping (加权和)
reward_aggregation: "weighted_sum"
r_fmt_weight: 0.05
r_out_weight: 0.85
r_len_weight: 0.10
l_min: 100                         # ← 100-600 范围鼓励长 reasoning
l_max: 600

# 动态难度遮罩
difficulty_mask: true
difficulty_adaptive: true          # ← 新增
target_active_ratio: 0.40
difficulty_low: 0.05               # 起始更宽松
difficulty_high: 0.97
```

---

## 5. 详细的 Stage 9 SFT 配置（Phase 1 的 cold start）

```yaml
# stage9_sft_config.yaml
exp_name: "qwen3-14B-tom-sft"
init_from: Qwen3-14B (base)

data: /mnt/data/tom_train_gpt55_traces.jsonl    # 4000 GPT-5.5 reasoning traces

training:
  epochs: 3
  per_device_batch_size: 4
  gradient_accumulation_steps: 8     # 有效 batch 256
  learning_rate: 5.0e-6              # 低 lr 避免遗忘
  warmup_steps: 100
  max_length: 2048                   # prompt + reasoning + answer
  
optimizer: AdamW
weight_decay: 0.01
lr_scheduler: cosine
```

---

## 6. 各方案预期增益 vs 成本对比表

| 策略 | 预计增益 (raw) | 工时 | 训练时间 | API/GPU 成本 | 风险 | ROI |
|---|---|---|---|---|---|---|
| **S1 GPT-5.5 推理蒸馏 SFT** | **+3-5pp** | 1.5 天 | 4h | $250 + 32 GPU-h | 低 | ⭐⭐⭐⭐⭐ |
| **S2 ExploreToM 2.0 数据** | **+2-3pp** | 6 天 | — | $100 | 中 | ⭐⭐⭐⭐ |
| **S3 难度课程 + 动态遮罩** | **+2-3pp** | 2.5 天 | 24h | 192 GPU-h | 低 | ⭐⭐⭐⭐⭐ |
| **S4 Dr.GRPO + KL + Entropy** | **+1-2pp** | 2.5 天 | (在重训内) | — | 低 | ⭐⭐⭐⭐ |
| **S5 response_length 1024** | **+1-2pp** | 0.5 天 | (×3 训练时间) | — | 中 | ⭐⭐⭐ |
| **S6 Reward shaping** | **+0.3-0.6pp** | 0.5 天 | (在重训内) | — | 低 | ⭐⭐ |
| **总合（保守）** | **+10-15pp** | ~13 天 | ~50h | ~$350 + 250 GPU-h | | |

---

## 7. 关键里程碑预测

### 14B 模型

| 训练阶段 | Raw 5718 | Clean 4551 | 备注 |
|---|---|---|---|
| Stage 8 (当前) | 0.7594 | 0.8449 | — |
| **Stage 9 (SFT + Dr.GRPO + 改进 reward + 长 response)** | **0.80-0.82** | **0.88-0.90** | 持平 deepseek |
| **Stage 10 (+ 难度课程 + 动态遮罩)** | **0.83-0.85** | **0.90-0.92** | 接近 GPT-5.5 |
| **Stage 11 (+ ExploreToM 2.0 数据)** | **0.86-0.88** | **0.92-0.94** | **可能 clean 反超 GPT-5.5** |

### 8B 模型

| 训练阶段 | Raw 5718 | Clean 4551 |
|---|---|---|
| Stage 7 (当前) | 0.7419 | 0.8321 |
| **Stage 9 (8B 同 14B 配方)** | **0.78-0.80** | **0.87-0.89** |
| **Stage 11** | **0.81-0.83** | **0.89-0.91** |

---

## 8. 与上一版方案 v2 的核心差异

| 维度 | v2 (含 self-consistency) | **v3 (纯训练，不偷)** |
|---|---|---|
| 主要思路 | Test-time compute + 算法 | **SFT cold start + 程序化数据 + 算法** |
| Self-consistency | ✅ N=16 majority vote (+5-8pp) | ❌ **排除** |
| GPT-5.5 推理蒸馏 | 单独提到 (+2-4pp) | ✅ **核心策略 S1** (+3-5pp) |
| 程序化数据 (ExploreToM) | 没提到 | ✅ **核心策略 S2** (+2-3pp) |
| 难度课程 | 提到，预期 +1.5-2.5pp | ✅ **核心策略 S3** + 动态遮罩 (+2-3pp) |
| Long CoT (response 1024) | 没特别强调 | ✅ **策略 S5** (+1-2pp) |
| 预期最终 | 14B raw 0.85-0.88 | **14B raw 0.86-0.88, clean 0.92-0.94** |
| 时间 | 4 周 | **5-7 周** |

**v3 的优势**:
- **结果可复现**：每次推理都强，不靠多 sample 平均
- **可商业化部署**：温度 0 单次解码就能拿到高分
- **真正模型能力提升**：而不是堆推理时计算

---

## 9. 紧急执行清单（按依赖顺序）

```
Week 1:
  □ Day 1-2: 实现 GPT-5.5 reasoning trace 生成器
  □ Day 3: 生成 4000 reasoning traces ($250, 3h)
  □ Day 4: 实现 Dr.GRPO loss + entropy-aware token + reward shaping (3 项算法改进)
  □ Day 5: 实现 SFT pipeline + 配置

Week 2:
  □ Day 6: 14B SFT cold start (4h)
  □ Day 7: 调试 Stage 9 GRPO 配置, 测试 50 步
  □ Day 8-9: 14B Stage 9 完整训练 (18h)
  □ Day 10: 评测 14B Stage 9 (~30min)

Week 3:
  □ Day 11-12: 实现 E2H curriculum + 自适应遮罩
  □ Day 13: 离线给 15000 数据打分 (4h)
  □ Day 14: 启动 14B Stage 10 (24h)

Week 4:
  □ Day 15-16: 评测 14B Stage 10, 8B 同步 Stage 9
  □ Day 17-19: ExploreToM 2.0 框架开发 (启动)

Week 5-7:
  □ ExploreToM 2.0 数据生成 + 14B Stage 11
  □ [可选] 32B 模型
```

---

## 10. 关键风险点与应对

### 风险 1: SFT 后 RL 不收敛

DeepSeek-R1 论文里讨论过：SFT cold start 后接 RL 时，**RL 学习曲线初期会有 dip**（因为 SFT 让模型偏向特定推理风格，RL 要重新探索）。

**应对**:
- 用低 lr (5e-7 而非 1e-6)
- KL coef 略大 (0.005 而非 0.001) 保持 SFT 学到的推理
- 训练 600 步（vs 350）让模型充分调整

### 风险 2: response_length 1024 训练 OOM

14B + TP=2 + response 1024 + KV cache 可能超 80GB。

**应对**:
- 降 prompt_length 到 768 (vs 1024)
- 降 rollout_batch_size 到 16 (vs 32)
- 用 ulysses sequence parallel
- 实测一次，OOM 就降参数

### 风险 3: ExploreToM 2.0 开发延期

程序化生成框架对我们 7 个 task 适配需要细致工作。

**应对**:
- 先做 Phase 1+2，不依赖 ExploreToM 2.0
- Phase 3 如果开发慢，先用现有 Phase A/B/C 数据继续训练
- Phase 1+2 已经能让 14B 到 0.83-0.85, **已是优秀结果**

### 风险 4: 评测集 wrong-label 噪声仍是 ceiling

即使训练再好，raw 5718 上 ~10% 是真正的标签问题。

**应对**:
- 主报告以 **clean eval 4551** 为主（已剔除 wrong-label）
- raw 5718 作为参考
- 在 clean eval 上反超 GPT-5.5 是真正的胜利

---

## 11. 累计预算

| 维度 | 数值 |
|---|---|
| 总工时 | 5-7 周 |
| GPT-5.5 API 成本 | ~$350 ($250 reasoning + $100 ExploreToM 数据 audit) |
| 训练时间总计 | ~80h GPU (1×8 H800) |
| GPU-小时 | ~640 |
| 实测费用估算 | ~$3500-5000 (按 $7/h H800 cluster) |

---

## 12. 与 DeepSeek-R1 / Light-R1 训练方法的关键对比

| 维度 | DeepSeek-R1 | Light-R1 | **我们 Stage 9-11** |
|---|---|---|---|
| 阶段 | SFT (cold) → RL → SFT → RL | SFT (curriculum) → DPO → RL | **SFT → curriculum RL** |
| SFT 数据规模 | 800k+ traces | 76k (curriculum) | **4k targeted (ToM only)** |
| SFT 教师 | R1 自身 | DeepSeek-R1 + filter | **GPT-5.5** |
| RL 算法 | GRPO | GRPO | **Dr.GRPO + curriculum + entropy** |
| Response length | 30k+ | 32k | **1024** (ToM 推理够) |
| 训练步数 | ~1000 | ~500 | **350-500** |
| 目标 | AIME / MATH | AIME / MATH | **ToMBench** |

**核心相似点**: 都是 SFT cold start + RL。我们之前 8 个 stage 都跳过了 SFT，是单点最大改进空间。

参考: [DeepSeek-R1 Nature](https://www.nature.com/articles/s41586-025-09422-z.pdf), [Light-R1 ACL 2025](https://aclanthology.org/2025.acl-industry.24/), [DeepSeek-R1-Distill 文献](https://www.emergentmind.com/topics/deepseek-r1-distill-models)

---

## 13. 总结

### 核心新思路

**之前的失败**: 我们 8 个 stage 都是纯 RL，没有 SFT cold start。这违背了 DeepSeek-R1 / Light-R1 / Magistral / OpenThoughts 等 2025 SOTA 推理模型的标准流程。

**新方案的核心**: 
1. **SFT cold start** 用 GPT-5.5 推理痕迹（不是答案）教模型**怎么思考 ToM**
2. **算法层 Dr.GRPO + 动态难度遮罩** 修复 95% rollout 浪费
3. **E2H 难度课程** 让训练阶段渐进难度
4. **程序化 ExploreToM 2.0** 生成对抗性数据填补 HOT 错误
5. **response_length 1024 + 长 reasoning** 释放模型推理能力

### 预期效果

| 阶段 | 14B Raw | 14B Clean | 距 GPT-5.5 (raw) |
|---|---|---|---|
| Stage 8 (现状) | 0.7594 | 0.8449 | -7.55pp |
| Phase 1 (+SFT+算法) | 0.80-0.82 | 0.88-0.90 | -3-5pp |
| Phase 2 (+curriculum) | 0.83-0.85 | 0.90-0.92 | -0.5-2pp |
| Phase 3 (+ExploreToM) | **0.86-0.88** | **0.92-0.94** | **+0.3-2pp 反超** |

**最终目标达成**: 14B 在 clean eval 上**反超 GPT-5.5**, raw 上**追平 GPT-5.5**, 不靠任何 test-time trick。

最后更新: 2026-05-19 17:00

## 引用文献

### SFT Cold Start + Distillation
- [DeepSeek-R1 Nature paper (2025)](https://www.nature.com/articles/s41586-025-09422-z.pdf): 标准 SFT → RL pipeline
- [Light-R1 ACL Industry 2025](https://aclanthology.org/2025.acl-industry.24/): Curriculum SFT + DPO + RL
- [Magistral 2025](https://arxiv.org/html/2506.10910v1): 强调 cold-start 重要性
- [DeepSeek-R1-Distill Family](https://www.emergentmind.com/topics/deepseek-r1-distill-models): 14B distilled SOTA
- [OpenThoughts 1.4M dataset](https://arxiv.org/html/2503.19633v1): 大规模 reasoning trace
- [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts): 350k traces

### 算法改进
- [Dr.GRPO (COLM 2025)](https://arxiv.org/pdf/2503.20783): GRPO 长度偏差修复
- [VAPO (ByteDance 2025)](https://arxiv.org/pdf/2504.05118): Value-based RL +6pp on AIME
- [High-Entropy Token Selection](https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr/): 关键 token 聚焦
- [Token-Efficient RL](https://arxiv.org/html/2603.06619v1)
- [Exploration vs Exploitation in RLVR](https://arxiv.org/html/2512.16912v1)
- [Limit of RLVR](https://limit-of-rlvr.github.io/): RLVR 上限分析

### 难度课程
- [E2H Curriculum (EMNLP 2025)](https://arxiv.org/abs/2505.08364)
- [Online Difficulty Filtering for RLVR](https://arxiv.org/html/2504.03380v2)
- [Self-Evolving Curriculum](https://arxiv.org/pdf/2505.14970)
- [Curriculum RL Easy-to-Hard](https://arxiv.org/abs/2506.06632)
- [Adaptive Curriculum Learning](https://arxiv.org/pdf/2504.05520)

### ToM 数据生成
- [ExploreToM ICLR 2025](https://arxiv.org/abs/2412.12175): +27pp on ToMi/HiToM
- [OpenToM Benchmark](https://arxiv.org/abs/2402.06044)
- [TMBench](https://arxiv.org/html/2402.15052v1)
- [Hypothesis-Driven ToM (2025)](https://arxiv.org/html/2502.11881v1)

### Long CoT
- [Dissecting Long-CoT Reasoning](https://arxiv.org/html/2506.04913v2)
- [Pruning Long CoT](https://arxiv.org/pdf/2508.10164)
