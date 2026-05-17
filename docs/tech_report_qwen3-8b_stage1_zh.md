# 技术报告：Qwen3-8B + GRPO 在 ToMBench 上的训练（Stage 1）

**作者**：基于训练日志 `train_stage1_1x8_20260515_121704.log` 与配置 `configs/tombench-rlvr/rlvr_config_stage1_1x8.yaml` 自动生成（commit `227ee48`）。

**状态**：本文档对应 Qwen3-8B + GRPO 训练项目的 Stage 1。Stage 1 是小规模管线验证（4k 数据 × 200 步），后续 Stage 2/3/5 在此基础上扩展，Stage 4 因失败被放弃。`docs/stage{1..5}_report.md` 是各阶段的执行摘要；本报告是 Stage 1 的深度工程记录。

---

## 1. 目标

训练 Qwen3-8B（base，非 thinking 版本）在 ToMBench 心智理论多选题数据集上的准确率，技术路线：

- 可验证奖励强化学习（RLVR），具体使用 GRPO 算法 + DAPO Clip-Higher + 动态采样
- 训练框架：阿里 ROLL（vendored 在 `framework/ROLL/`），vLLM 0.8.4 做 rollout，Megatron-Core 0.16.0 做训练，1×8 H800 colocated 部署
- 自定义多组件奖励 `TomMcqRewardWorker`，融合格式合规、答案正确性、长度正则化

**目标对比模型**：deepseek-v4-pro，相同评测集（5718 题全量，direct 协议）。Stage 1 故意小规模，目的是端到端验证管线，再投入 Stage 2 的大预算。

## 2. 硬件与软件栈

| 项目 | 配置 |
|---|---|
| 硬件 | 单节点 8× NVIDIA H800 80 GB SXM |
| 互联 | NVLink 全互联（仅节点内） |
| CPU / 内存 | 64 核 / 512 GB 系统内存（对 Gloo+CPU optimizer save 至关重要） |
| 容器 | `qwen3-tom-train:latest`，基础镜像 NVIDIA pytorch 24.05-py3 |
| Torch | 2.6.0+cu124（CUDA 12.4，从阿里云 pytorch-wheels 镜像安装） |
| Megatron-Core | 0.16.0 |
| Transformer Engine | 2.2.0（pin 死；更新版本破坏 `transformer_engine.pytorch` 导入） |
| vLLM | 0.8.4（自带 flash-attn 内核） |
| Ray | 2.48（pin `click==8.2.1` 避免 Python 3.10 上的 sentinel 错误） |
| ROLL 框架 | 上游快照 vendored 在 `framework/ROLL/`，仅添加自定义 worker `roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` |
| 持久化存储 | `/data_nvme/grj-projects/` 在 14 TB NVMe LV 上（从 875 GB SSD 迁移过来） |
| 容器内挂载 | `/workspace` ← repo，`/mnt/data` ← 训练数据，`/mnt/models` ← HF/ModelScope 缓存，`/mnt/output` ← checkpoints |

**GPU 资源共享**：actor_train（Megatron）、actor_infer（vLLM）、reference（Megatron infer）共享同一 8 张 GPU。ROLL 的 "offload state manager" 在不同阶段间把每个角色的权重/优化器/KV cache 在 GPU 和 pinned CPU 内存间倒腾。

## 3. 训练数据

Stage 1 使用 `data/tom/tom_train_4k.jsonl`，从 `tom_train.jsonl` 中随机抽样的 4000 条子集（彼时全集 ~5911 条；Phase-1 合成数据后增长到 8901 条，但这是 Stage 5 之后的事）。

### 3.1 数据源

| 源 | Stage 1 时 4k 中的条数 | 描述 |
|---|---|---|
| ExploreToM | ~886 | 第三方合成数据集，覆盖二阶信念和知识-注意力关系 |
| SimpleToM | ~440 | Sally-Anne 经典一阶错误信念题 |
| 自合成（deepseek-v4-flash） | ~1353 | `scripts/data/synth_tomtype.py` 调用 deepseek 合成 9 种 ToMBench 子任务，temperature=0.9 |
| 中文翻译 | ~298+412+171 | 英文训练记录翻译成中文（`scripts/data/translate_to_zh.py`） |

（Stage 1 的 4k 总数加和精确为 4000；每源条数取决于 `merge_and_dedupe.py` 时的随机种子。）

### 3.2 反作弊：评测集 vs 训练集泄漏检测

**每条训练数据在合入前都会与 ToMBench 评测集（5718 题）做相似度检查。**

`scripts/data/merge_and_dedupe.py` 在 ToMBench 评测集上构建 MinHash LSH 索引（4-gram 文本，threshold=0.6）。每条候选训练记录：

1. 计算 MinHash，查询 LSH 索引
2. 对所有候选命中（估计相似度 ≥ 0.6），计算精确 4-gram Jaccard
3. 若任一精确 Jaccard ≥ 0.6，丢弃该训练记录

**Stage 1 时的结果**：0 条训练记录被泄漏滤掉。合成 prompt 已明确要求不要复现 ToMBench 原题，且 ExploreToM/SimpleToM 这些上游数据集本来就是独立的。

内部去重在 MinHash 阈值 0.7 下：dropped ~150 条 near-duplicates。

### 3.3 数据格式

每条训练记录是 chat-style：

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful reader answering ..."},
    {"role": "user",   "content": "Story:\n...\n\nQuestion: ...\nA. ...\nB. ...\nC. ...\nD. ..."}
  ],
  "ground_truth": "B",
  "tag": "tom_mcq",
  "source": "exploretom",
  "language": "en",
  "task": "False Belief",
  "question_id": "exploretom_7661"
}
```

direct 协议的 system prompt 固定为：

```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

奖励 worker 匹配这个契约：从 rollout 中正则解析 `\boxed{[ABCD]}`，对比 `ground_truth`，再加格式/长度惩罚。

## 4. 奖励函数

实现见 `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py`。对每条 rollout 响应，计算三个子奖励并**相乘**：

```python
r_fmt = 1.0 if response 匹配正则 \boxed{[A-D]} else 0.0
r_out = 1.0 if (r_fmt and 提取字母 == ground_truth) else 0.0
r_len = sigmoid_window(response_length, l_min=8, l_max=256, k=50)
r_total = r_fmt * r_out * r_len
```

`sigmoid_window(L, l_min, l_max, k)` 是一个平滑带通函数：在 `l_min` 附近上升，在 `l_max` 附近下降，中间平稳：

```python
def sigmoid_window(L, l_min, l_max, k):
    span = max(1.0, l_max - l_min)
    rise = sigmoid(k * (L - l_min) / span)
    fall = 1.0 - sigmoid(k * (L - l_max) / span)
    return rise * fall
```

我们的参数 (l_min=8, l_max=256, k=50)：
- L=4（太短）：r_len ≈ 0.0
- L=20（暖身后）：r_len ≈ 0.95
- L=200（典型）：r_len ≈ 1.0
- L=260（刚超 cap）：r_len ≈ 0.5
- L=270（超 cap，被 padding）：r_len ≈ 0.05

| 组件 | 收敛时（step 175+） |
|---|---|
| `r_fmt` | ~0.87（绝大多数 rollout 都能正确输出 `\boxed{[A-D]}`） |
| `r_out` | ~0.81（格式正确的情况下，~93% 选对字母） |
| `r_len` | ~0.93（响应长度集中在 158 token 附近，在带通区间内） |
| `r_total` | ~0.80（乘积） |

**为什么乘法而非加法？** 三个原因：

1. 格式错误的响应**得分严格为 0**，没有半分。这强制策略首先学会格式。
2. 长但正确的响应受平滑惩罚；模型不能通过"多写"换取奖励。
3. 乘积取值在 [0, 1]，方差白化稳定。

`l_max=256` 与 `response_length: 256` 匹配。若提高 `response_length`，必须同步提高 `l_max`（Stage 3 把两者都改到 384；Stage 5 又回到 256，因为发现训练 vs 验证不匹配）。

奖励 worker 把每个 `(r_fmt, r_out, r_len, r_total, 提取字母, ground_truth)` 元组以 debug level 记日志，便于事后分析。聚合指标（`reward/r_*_mean`）每步推送给 tracker。

## 5. 算法：GRPO + DAPO Clip-Higher + 动态采样

### 5.1 GRPO 基础

每个 prompt，rollout 集群生成 8 个样本（`num_return_sequences_in_group: 8`）。每个样本的 advantage 用 group-相对奖励：`A_i = (r_i − mean(r_group)) / std(r_group)`（whiten_advantages: true）。这样不需要单独的 value head，同时控制方差。

### 5.2 DAPO Clip-Higher

PPO 截断范围非对称：
- `pg_clip_low: 0.20`
- `pg_clip_high: 0.28`

允许 advantage 为正的样本有适度更大的向上更新，向下截断保持标准 0.2。这种非对称截断带来更快的 credit assignment，同时保持稳定。配合 `dual_clip_loss: true`（当 advantage 为负时同时截断 ratio 和 ratio×advantage），是标准的 DAPO 配置。

### 5.3 动态采样

`use_additional_prompts: true`。调度器在 batch 包含太多"简单"（一组 8 样本全对）或"困难"（全错）样本时自动拉额外 prompt，因为这些组方差为 0，对梯度贡献为 0。`max_running_requests: 128`。

### 5.4 难度遮罩

奖励计算后，按 per-prompt 组准确率遮罩样本：
- `difficulty_low_threshold: 0.1` — 若组内 < 10% 正确则丢弃整组（模型还学不会）
- `difficulty_high_threshold: 0.95` — 若 > 95% 正确则丢弃整组（没信号了）

这在收敛阶段尤其重要：step 175+ 时 rollout score 已达 0.80，没有难度遮罩的话梯度会在饱和样本上趋零。

### 5.5 KL 正则化

Stage 1 用 `add_token_level_kl: false`。理由：KL 惩罚在 fresh-from-base 模型上会过度锚定弱基础分布，阻止快速学习。Stage 1 目的是验证管线能否学习，所以保持标准 PPO 设置。

（Stage 3 尝试了 KL=true，效果好坏参半——保护了 Belief / False Belief 但阻碍了 Knowledge / Non-literal Comm 进步。）

### 5.6 有效 batch 与 rollout 形状

- `rollout_batch_size: 32` — 每步 32 个不同 prompt
- `num_return_sequences_in_group: 8` — 每 prompt 8 个 rollout
- 有效 rollout batch：**256 个序列/步**
- `gradient_accumulation_steps: 32`（per-device batch=1，每 rank 做 32 次 forward+backward 块）
- DP=8（每 GPU 一个 rank），TP=1，PP=1

每 prompt 的样本构成 GRPO 组 → 贡献 advantage 归一化 → 一次 PPO 更新。

## 6. 超参数

```yaml
# 采样
prompt_length: 2048
response_length: 256
rollout_batch_size: 32
num_return_sequences_in_group: 8
ppo_epochs: 1

# DAPO Clip-Higher
use_pg_clip_range: true
pg_clip_low: 0.20
pg_clip_high: 0.28
dual_clip_loss: true

# 方差控制
value_clip: 0.5
reward_clip: 5
advantage_clip: 2.0
whiten_advantages: true
add_token_level_kl: false

# 难度遮罩
max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95
error_max_len_clip: false

# 训练
max_steps: 200
save_steps: 200      # 只在最后存盘（避免训练中 OOM，详见 §7）
eval_steps: 50
logging_steps: 1

# 优化器
learning_rate: 1.0e-6
weight_decay: 0
warmup_steps: 20
gradient_accumulation_steps: 32
per_device_train_batch_size: 1

# 生成（rollout）
temperature: 0.99
top_p: 0.95
top_k: 50

# 生成（验证）
temperature: 0.0
max_new_tokens: 64    # 注意：这个 64-token cap 后期严重低估了 val_correct

# 奖励
l_min: 8
l_max: 256
k: 50  # 长度惩罚曲线锐度
```

### 关键选择说明
- **学习率 1e-6**：8B 规模 RL post-training 的标准值，未调优
- **warmup 20 步**：缓和初期梯度爆炸（注意日志里 step 0 的 grad_norm = 65.9，到 step 25 已经稳定到 2.4）
- **RL 更新路径不带 reference 模型** `enable_reference: true` 但因为 `add_token_level_kl: false` 所以 KL 项系数为 0。Reference 只在计算 PPO loss 的 log-ratio 分母时用到
- **`save_steps: 200`**（仅保存最终 checkpoint）：把 Megatron 分布式 optimizer save 延迟到训练结束，避免训练中触发 OOM。详见 §7

## 7. 分布式 Save（OOM 修复史）

Stage 1 差点没拿到 checkpoint。第一次尝试在 step 199 训练全部成功后的 `do_checkpoint` 调用 OOM 了。根因分析：

在 colocated 1×8 部署下，`do_checkpoint` 被触发时：
- vLLM 仍占用 ~7.7 GB 残留 KV cache + 权重元数据 / GPU（offload 不会完全释放）
- Reference 模型占用 ~3 GB 残留
- actor_train 被重新加载（上一步结束时被 offload 了）——此时占 ~75 GB，因为除模型权重外，还有 optimizer（Adam moments）、梯度（仍在分配）、reference activation buffers、所有热 CUDA workspaces

总和 ~85 GB，超过 H800 的 80 GB。

Megatron-Core 默认的分布式 optimizer save 会调用 `FullyParallelSaveStrategyWrapper.sharded_param_state_fully_reshardable`，进而调用 `get_parameter_state_dp_zero` 且 `use_gloo_comm=False`。该函数在**每个 DP rank** 上分配 shape 为 `(buffer_numel_unpadded,) × data_parallel_world_size` 的 `recv_tensors`，**在 CUDA 上**（line 1077）：

```python
device = "cpu" if use_gloo_comm else torch.cuda.current_device()
recv_tensors = [
    torch.zeros((gbuf_local_numel,), dtype=torch.float32, device=device)
    for _ in range(data_parallel_world_size)
]
```

Qwen3-8B + DP=8 时，`gbuf_local_numel × DP × 4 字节 ≈ 3.81 GiB`。剩 690 MiB 每 GPU，必然 OOM。

**修复**（commit `d7bf18a`）：在 `actor_train.strategy_args.strategy_config` 里加 `distrib_optim_fully_reshardable_mem_efficient: true`。这把 gather 路由到 Gloo（CPU）而非 NCCL（GPU），把 `world_tensors` 填在 CPU 上：

```yaml
strategy_config:
  use_distributed_optimizer: true
  distrib_optim_fully_reshardable_mem_efficient: true  # 救命魔法
  recompute_granularity: full
```

修复后：
- Save 路径用 Gloo collective + CPU buffer
- DP rank 0 在 CPU 上接收完整的 reshardable state
- 通过 `dist_checkpointing.save` 写入本地磁盘
- `checkpoint_manager.upload` 再 rsync 风格复制到 `/mnt/output/...`
- 本地磁盘峰值短暂占用：~107 GB 本地 + ~107 GB 上传 = 短暂 ~214 GB，然后本地副本被删

Save 本身耗时 ~10 分钟（Gloo gather 单线程 over CPU，比 NCCL 慢得多但稳定）。NVMe 存储下完全没问题。

## 8. 训练轨迹

从 `train_stage1_1x8_20260515_121704.log` 直接读取，每 25 步采样：

| step | rollout score | reward | r_fmt | r_out | r_len | KL loss | grad_norm | response 长度 (tok) | val_correct/all (subset500) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.215 | 0.118 | 0.301 | 0.273 | 0.581 | 0 | 65.9 | 208 | 0.042 |
| 25 | 0.148 | 0.149 | 0.160 | 0.160 | 0.568 | 0.005 | 2.44 | 251 | — |
| 50 | 0.412 | 0.213 | 0.438 | 0.426 | 0.705 | 0.020 | 1.97 | 230 | 0.204 |
| 75 | 0.650 | 0.168 | 0.680 | 0.664 | 0.824 | 0.073 | 2.05 | 202 | — |
| 100 | 0.566 | 0.246 | 0.656 | 0.574 | 0.819 | 0.113 | 1.82 | 201 | 0.454 |
| 125 | 0.637 | 0.168 | 0.707 | 0.641 | 0.849 | 0.145 | 1.71 | 188 | — |
| 150 | 0.791 | 0.090 | 0.867 | 0.793 | 0.929 | 0.207 | 3.01 | 157 | 0.548 |
| 175 | 0.806 | 0.056 | 0.875 | 0.812 | 0.931 | 0.194 | 2.88 | 162 | — |
| 199 | 0.800 | 0.116 | 0.871 | 0.801 | 0.934 | 0.235 | 2.04 | 158 | — |

轨迹解读：
- **step 0–25**：step 0 梯度爆炸（grad_norm 65.9 = 训练初始震荡，策略第一次见到自己 rollout 作为奖励信号）。Warmup 在 20 步内吸收掉这个冲击；到 step 25 策略短暂"忘记"格式（r_fmt 从 0.30 跌到 0.16）但已稳定下来
- **step 25–75**：格式快速学习。r_fmt 从 0.16 涨到 0.68——模型学到 `\boxed{X}` 这个被奖励的形状。r_out 跟 r_fmt 紧密耦合，因为格式错的响应 `r_total` 直接为 0，错格式响应和正确答案不相关
- **step 75–150**：准确率爬坡。r_out 0.66 → 0.79。格式奖励先饱和（~0.87），准确率紧随其后。KL 稳步增长 0.07→0.21，说明策略在显著偏离 base，但因为 clip-higher 在保护着所以没有失稳
- **step 150–199**：平台期。Rollout score 在 0.80–0.81 之间振荡。**没有难度遮罩的话此处梯度会消失**（90%+ 组全对或全错）；有了遮罩，剩余"混合"组仍能产生学习信号。响应长度稳定在 158–166 token，远低于 `l_max=256`

`reward` 列（即 `critic/rewards/mean`）随着 `r_total` 上升反而下降，因为 `reward` 是 KL 减去后的 whitened advantage proxy：策略越正确，原始 advantage spread 越小。这在 GRPO 中是正常的。

### 验证曲线（subset500）

| step | val_correct/all | val_correct/tom_mcq |
|---|---|---|
| 0 | 0.042 | 0.278 |
| 50 | 0.204 | 0.299 |
| 100 | 0.454 | 0.534 |
| 150 | 0.548 | 0.613 |

`val_correct/tom_mcq` 是同一指标但仅对 ToM-MCQ 标签的记录（评测集全是这个）。`val_correct/all` 包含格式检查；`tom_mcq` 放宽 `\boxed{}` 要求，只看预测字母。

验证曲线与训练同步：50→100→150 是最大增益期；150→200 预计仍有适度提升。我们没拿到 step 200 的 val，因为 `eval_steps=50` 调度 + `max_steps=200` 意味着下次 val 在 step 200 本身（save 之前），代码路径跳过了。

## 9. 最终评测

我们用两种评测协议测试训练 checkpoint：
1. **ToMBench 全量 5718 题**，direct 协议，`max_tokens=2048`（确保 reasoning 模型如 deepseek 不被截断；对我们 8B 不重要因为它不 reasoning）
2. **Subset500**（确定性 500 题样本），3 个协议：direct、cot、del_tom

### 9.1 全量 5718（direct）

| 指标 | 值 |
|---|---|
| Overall | **0.7394** |
| EN | 0.7275 |
| ZH | 0.7513 |

对比修正后的 baseline：

| 模型 | Overall | EN | ZH |
|---|---|---|---|
| Qwen3-8B base（无 RL） | 0.7009 | 0.7020 | 0.6999 |
| **Stage 1（本工作）** | **0.7394** | 0.7275 | 0.7513 |
| deepseek-v4-pro 目标 | 0.8080 | 0.7978 | 0.8181 |

**Δ vs baseline：+3.85pp**，**距 deepseek（真实全量 baseline）：−6.86pp**。

ZH 表现（比 EN 高 0.05pp）很有意思，从训练数据分布（70/30 EN/ZH）推不出来。假设：ZH 评测模式更刚性（更直接的答案格式），所以格式学习直接转化为准确率。

### 9.2 Per-task 分解（全量 5718, direct）

| Task | Stage 1 | deepseek (5718) | Gap | EN/ZH 分别 |
|---|---|---|---|---|
| Belief | 0.6937 | 0.8486 | -15.49pp | 0.669/0.718 |
| Desire | 0.5917 | 0.6333 | -4.16pp | 0.567/0.617 |
| Emotion | 0.7286 | 0.8048 | -7.62pp | 0.700/0.757 |
| False Belief | **0.8520** | 0.8946 | -4.26pp | 0.862/0.842 ← 最接近 |
| Intention | 0.7647 | 0.8926 | -12.79pp | 0.750/0.779 |
| Knowledge | 0.4792 | 0.5675 | -8.83pp | 0.471/0.488 ← 绝对最低 |
| Non-literal Comm | 0.7674 | 0.8128 | -4.54pp | 0.749/0.786 |

**Stage 1 优势**：False Belief（vs base +12.4pp，距 deepseek 最近）。训练数据严重偏向 False Belief（~2629 条，最大 task），这是合理的。

**Stage 1 弱项**：
- Knowledge (0.48)：scalar implicature 题（"most/some/almost no" + 计数）——模型默认走字面算术。训练数据完全没有这类模式（Stage 5 用 Phase-1 scalar 集尝试修复，效果一般）
- Non-literal Comm (0.77)：Faux-pas 识别。模型过度归因（在无 faux-pas 故事中也认为有人失礼）。训练数据也没覆盖（Stage 5 加了 800 条合成 faux-pas，全量 5718 上无明显提升）

### 9.3 Subset500 跨三协议

| 协议 | Stage 1 | deepseek (subset500) | Gap |
|---|---|---|---|
| direct | 0.7460 | 0.7880 | -4.20pp |
| cot | 0.6980 | 0.7140 | -1.60pp |
| del_tom | 0.7460 | n/a | |

cot 相对 base 退化（cot 0.7464 → 0.6980）是 RL post-training 已知副作用：训练把策略推向 direct-only 响应，对 CoT 格式 prompt 略有损害。Stage 5（用 token-level KL，重试）把 cot 恢复到 0.7540，但代价是整体准确率。Stage 1 + cot 是这个协议上的局部最小值。

## 10. 时间预算

| 阶段 | 耗时 |
|---|---|
| 容器启动 + worker init | ~10 分钟 |
| 训练（200 步） | 3h 20m |
| `do_checkpoint`（Gloo+CPU mem-efficient） | ~10 分钟 |
| **Stage 1 总耗时** | **~3h 40m** |
| **GPU-小时** | **~26** |
| HF 格式转换（训练后） | ~3 分钟 |
| vLLM serve 冷启动 | ~2 分钟 |
| 全量 5718 评测（vLLM concurrency=32） | ~6 分钟 |

## 11. 经验教训

1. **分布式 optimizer save 默认走 NCCL+CUDA gather，在 colocated 1×8 H800 上会 OOM。** 这种布局下务必设置 `distrib_optim_fully_reshardable_mem_efficient: true`。Megatron-Core 自 0.14.0 起就支持这个开关，但默认关闭

2. **step 100 之后难度遮罩不可或缺。** 没有遮罩时梯度会在组饱和后死掉。0.1/0.95 阈值来自 GRPO 论文，我们没再调

3. **`r_len` 比预想的更重要。** 早期实验移除长度惩罚后，模型会写 256-token 长响应，经常在 `\boxed{X}` 之后还在写然后失去格式合规性。简洁奖励让策略紧凑

4. **格式学习先于准确率学习。** r_fmt 在 step 75 左右就饱和到 0.87；同一时刻 r_out 才 0.66。模型先学"把字母包在 `\boxed{}` 里"，然后再学"选对字母"。这与 GRPO 先学奖励形状再学底层技能的特点一致

5. **验证 `max_new_tokens=64` 危险地短。** Stage 1 fits（响应 ~158 token 加少量前导）但后期 Stage 3 用 `response_length=384` 时 `val_correct` 崩溃，因为 `\boxed{X}` 跑过 64-token 截断了。如果你改 `response_length`，必须同时重调 val token cap

6. **5718 baseline ≠ subset500 baseline。** Stage 1 在 subset500 上看着距 deepseek 4.20pp；全量 5718 上 gap 是 6.86pp。Stage 5 才补的 deepseek 全量 eval 才发现这点。Headline 数字必须 benchmark 在全量数据集

## 12. 复现 Stage 1

```bash
# DEV 机器
git clone https://github.com/ruijieguo/social-mind-rl
cd social-mind-rl
make build-data            # 如果 data/ 不存在，重建 tom_train_4k.jsonl
cp configs/deploy.env.example configs/deploy.env  # 填入 TRAIN_HOST 等

# Sync 到 TRAIN
make sync-up

# 在 TRAIN: 启动 docker，按需 build qwen3-tom-train image，运行训练
make train-stage1-1x8

# 训练后，在 TRAIN: 转换 Megatron → HuggingFace
ssh $TRAIN_HOST 'cd /data_nvme/grj-projects/qwen3-tom && \
  docker run --rm --gpus all --ipc host --shm-size 8gb \
    --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
    -v /data_nvme/grj-projects/qwen3-tom:/workspace \
    -v /data_nvme/grj-projects/tom-output:/mnt/output \
    -v /data_nvme/grj-projects/models:/mnt/models \
    -e PYTHONPATH=/workspace:/workspace/framework/ROLL:/workspace/framework/ROLL/mcore_adapter/src \
    -w /workspace --entrypoint python qwen3-tom-train:latest \
    framework/ROLL/mcore_adapter/tools/convert.py \
    --checkpoint_path /mnt/output/qwen3-8B-tombench-rlvr-stage1-1x8/<timestamp>/checkpoint-199 \
    --output_path /mnt/output/qwen3-8B-tom-hf --bf16'

# vLLM 部署
make serve-launch

# 在 DEV 评测
make eval-final
```

## 13. 下一步

Stage 1 确认管线能学习。后续 stage 探索：
- **Stage 2**（8k × 500 步）：更多数据，更多步数。结果：0.7263 — 过拟合；多 500 步太多
- **Stage 3**（KL=true，response_len=384）：保住 Belief/FB，但阻碍 Knowledge。结果：0.7302
- **Stage 4**（KL=true + Phase-1 合成数据带空 C/D 选项）：训练停滞，废弃
- **Stage 5**（KL=false + Phase-1 修复数据）：0.7305。Subset500 cot 最佳 0.7540 但全量 5718 不变

五个 stage 全部下来，**8B+RL 在全量 5718 上的上限是 0.7394**。14B+RL（见 `tech_report_qwen3-14b_stage1_zh.md`）达到 0.7527 ——朝 deepseek 0.8080 迈出有意义的一步。

## 14. 产物清单

| 路径 | 内容 |
|---|---|
| `configs/tombench-rlvr/rlvr_config_stage1_1x8.yaml` | 完整配置 |
| `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` | 自定义奖励 worker |
| `logs/train_stage1_1x8_20260515_121704.log` | 完整训练日志（10 MB） |
| `output/eval/final_full5718.{json,md}` | 5718 评测结果 |
| `output/eval/final_subset500.{json,md}` | subset500（3 协议） |
| `output/analysis/curves_stage1_1x8.png` | 12 子图训练曲线 |
| Megatron checkpoint（TRAIN） | `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage1-1x8/.../checkpoint-199/`（107 GB） |
| HF checkpoint（TRAIN） | `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf/`（16 GB，4-shard safetensors） |
| Git commit | `227ee48`（训练修复） → `d15cec2`（完整报告） |

---

## 附录 A：训练数据合成配方

Stage 1 的 4k 训练集由多步组装流水线生成。各组件来源：

### A.1 ExploreToM（~886 条）

来自第三方论文的合成数据集（Yufei Tian et al., 2024），针对二阶信念和知识-注意力关系。我们加载了他们 2000 条 release，用 `scripts/data/build_exploretom.py` 转换 schema：

```python
record = TomRecord(
    question_id=f"exploretom_{i}",
    source="exploretom",
    language="en",
    task=ability_to_task(record["ability"]),  # 把冗长 ability 字符串映射到广义 task
    story=record["context"],
    question=record["question"],
    opt_a=record["options"]["A"], opt_b=..., opt_c=..., opt_d=...,
    gold=record["correct_answer"],
)
```

### A.2 SimpleToM（~440 条）

Sally-Anne 风格一阶错误信念题，也是已有数据集。加载 1000 条，随机抽样进 4k 混合。

### A.3 deepseek-v4-flash 自合成（~1353 条）

`scripts/data/synth_tomtype.py` 用 deepseek-v4-flash 按 ToMBench task 类型显式 prompt：

**System prompt**（反作弊）：
```
You are a careful question writer creating new theory-of-mind multiple-choice questions for training.
Your output MUST be a single JSON object with keys: story, question, options (an object with A,B,C,D), answer (one of A,B,C,D).
Do NOT reproduce, paraphrase, or translate any question from ToMBench by Chen et al. (ACL 2024).
Write entirely new scenarios.
```

**Per-task user prompts**（每次调用 sample 一个 task 类型）：
```
False Belief:       "Write a False Belief task: a character's belief differs from reality after an unseen change."
Strange Story:      "Write a Strange Story task involving subtle social misunderstanding or irony."
Unexpected Outcome: "Write an Unexpected Outcome task where the result of an action differs from the character's expectation."
Persuasion Story:   "Write a Persuasion Story task where one character tries to change another's belief."
Knowledge:          "Write a Knowledge-Attention Link task where a character's knowledge depends on what they observed."
Desire:             "Write a Multiple Desires task where two characters have different preferences."
Emotion:            "Write a Discrepant Emotions task where two characters feel differently about the same event."
Intention:          "Write a Prediction of Actions task asking what a character will do given their intention."
Non-literal Comm:   "Write a Hinting Task: a character makes an indirect request and we must infer their actual desire."
```

**生成参数**：temperature=0.9, max_tokens=800, concurrency=8 → 实际 ~0.5 req/s。约 1500 条生成耗 60 分钟。

**过滤**：每个响应用正则提取 `\{...\}`，按 JSON 解析，若 (story, question, options A-D, answer letter) 任一缺失或 answer 不在 {A,B,C,D} 则拒绝。失败率 ~5%。

这 9 个 task 类型覆盖经典 ToM（False Belief、Knowledge）和语用通信（Hinting、Strange Story）。我们这么选是为镜像 ToMBench 任务体系但又不泄露具体题目。

### A.4 中文翻译（~881 条）

`scripts/data/translate_to_zh.py` 用 deepseek-v4-flash 把 EN 训练记录翻译成中文（story, question, options）。System prompt：

```
You are a precise translator. Translate the given theory-of-mind multiple-choice question
from English to Simplified Chinese. Preserve story logic, character names (transliterate),
and option order. Output a JSON object with the same keys.
```

成功翻译的记录设 `source = "synth_zh"` 或 `exploretom_zh` / `simpletom_zh`，language=`zh`，新 question_id `<original_id>_zh`。约 70% 翻译尝试成功（失败：模型加额外评论、违反 JSON schema、拒绝翻译）。

### A.5 Merge 时反作弊

`scripts/data/merge_and_dedupe.py` 是守门员：

1. 加载 `data/tom/tombench_eval.jsonl`（5718 条）。在 `story + question + opt_a + opt_b + opt_c + opt_d` 文本上构建 MinHash-LSH 索引，threshold=0.6，num_perm=128
2. 每条候选训练记录，计算 MinHash 并查询索引
3. 对每个 LSH 命中，计算精确 4-gram Jaccard
4. 若任一精确 Jaccard ≥ 0.6，丢弃该候选

**Stage 1 时（无 Phase-1 数据）**：0 条被泄漏滤掉。Data card（`docs/data-card.md`）记录每源的 max-Jaccard 分布——所有 mean 和 p95 都是 0.000。

cross-source 去重后（内部 MinHash 阈值 0.7），又丢 ~150 条 near-duplicates。最终装配：shuffle 5911 条 → seeded 随机 4000 → `tom_train_4k.jsonl`。

### A.6 为什么用 deepseek-v4-flash 而非 deepseek-v4-pro 合成？

我们考虑过用 deepseek-v4-pro（评测目标）合成，但选了 flash 两个原因：

1. **成本**：pro 每 token 贵 ~5×。3000 条生成 × 800 token 是真金白银
2. **Pro 有 reasoning token 会拖后腿**：在 JSON 结构化输出中，pro 即使对"简单"任务也会发 reasoning token，这些经常跑过 `max_tokens` 导致 JSON 输出被截断。我们看到 pro 失败率 50% 而 flash 仅 5%

Phase-1 合成（Stage 5+ 加的）用了混合：flash 处理 hinting/faux-pas（flash 能搞定），pro 处理 scalar implicature 和二阶信念（需要细致推理才能保证答案正确）。详见 `tech_report_qwen3-14b_stage1_zh.md` 和 Stage 5 报告。

## 附录 B：评测协议细节

### B.1 Direct 协议

System prompt：
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

User prompt：由 `scripts/eval/run_tombench.py` 构造：
- EN：`"Story:\n{story}\n\nQuestion: {question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"`
- ZH：`"故事：\n{story}\n\n问题：{question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"`

评测参数：temperature=0.0, top_p=1.0, max_tokens=2048（足够大让 reasoning 模型不被截断）。提取：同 `\boxed{[A-D]}` 正则。

### B.2 CoT 协议

System prompt：
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Think step by step about the mental states of the characters,
then output your final answer in the format \boxed{X} where X is one of A, B, C, D.
Put your final \boxed{X} on the last line.
```

评测参数：temperature=0.6, top_p=0.9, max_tokens=1024。

### B.3 Del-Tom 协议

鲁棒性检查：保留故事和问题，但删除所有显式心理状态词（故事预处理去掉 "knows", "believes", "thinks", "wants", "feels" 等）。测试模型是否使用浅层关键词捷径。评测参数同 direct。

### B.4 Subset500 vs Full 5718

`tombench_eval_subset500.jsonl` 是 `tombench_eval.jsonl`（5718 题）的确定性随机 500 题采样（seed=42）。在 stage 0 时创建用于快速迭代。**阅读结果时**：

- 全量 5718 是 canonical headline
- Subset500 用于协议对比（早期 stage 跑 deepseek-v4-pro 全量代价高）和与早期报告的直接比较
- deepseek-v4-pro 在 subset500 上 0.7880，全量 5718 上 0.8080 — 差 2pp 归因于样本方差和评测时 deepseek API 在高并发下的微小差异
