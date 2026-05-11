# Qwen3-8B ToMBench RLVR 后训练设计文档

**日期**：2026-05-11
**作者**：交互式 brainstorming 产出
**状态**：待用户审阅
**目标**：在 16×H800 上用 ROLL 框架对 Qwen3-8B 做 GRPO 后训练，提升 ToMBench 准确率，逼近或超越 deepseek-v4-pro

---

## §1 总体目标与成功标准

### 1.1 任务定义

在 ROLL 框架（`framework/ROLL/`）的 RLVR pipeline 上，对 `Qwen/Qwen3-8B` 做 GRPO 后训练，目标是提升其在 ToMBench（ACL 2024，Chen et al., 2860 题，8 大 ToM 任务，中英双语）上的准确率，逼近或超越 deepseek-v4-pro 在同一基准上的分数。

### 1.2 起点

- Qwen3-8B 在 ToMBench 上的起点分数 **Y_base_nt**（non-thinking 模式）和 **Y_base_t**（thinking 模式）**由我们亲自评测得到**（阶段 2），不依赖外部论文报告的 0.729 数字。
- 外部论文（arxiv 2602.10625 报告 Qwen3-8B non-thinking 0.729 / thinking 0.680；Social-R1 论文报告 Qwen3-8B disable thinking 0.5349 / thinking 0.6179）数字**仅作预期参照**，不作 baseline。

### 1.3 目标量化

- deepseek-v4-pro 在 ToMBench 上的分数 **X**（= `X_direct`，协议 1 主分数）由阶段 2 评测得到。
- 训练后 Qwen3-8B 分数 **Y'_direct**（best checkpoint 在协议 1 上的整体分数）。
- **逼近**：`Y'_direct ≥ X − 0.02`（ε = 2pp）
- **超越**：`Y'_direct ≥ X`
- 训练目标对标的 base 起点是 **Y_base_nt**（non-thinking 模式），因为训练时 system prompt 禁止解释性文字、`response_length=256`，与 non-thinking 行为一致。`Y_base_t` 仅作 thinking-vs-non-thinking 对照参考。

### 1.4 评测协议

三套协议并行：

- **协议 1 Direct（主分数）**：system prompt 要求只输出 `\boxed{X}`；规则抽取首个匹配
- **协议 2 CoT（参考分数）**：允许先思考再输出 `\boxed{X}`；规则抽取末个匹配
- **协议 3 DEL-ToM 增强（可选）**：仅 belief 类子任务，N=8 采样多数投票

**主分数为协议 1**，"逼近/超越"判据基于 `Y'_direct` 与 `X_direct`。

### 1.5 评测汇报维度

- ToMBench 整体 overall（八任务平均）
- 八个子任务分项
- 中文 / 英文两语种分别
- direct / cot / del_tom 三协议

### 1.6 对照组

| 角色 | 来源 |
|---|---|
| Qwen3-8B base (non-thinking) | DashScope API `qwen3-8b` + `enable_thinking=false` |
| Qwen3-8B base (thinking) | DashScope API `qwen3-8b` + `enable_thinking=true` |
| deepseek-v4-pro | DeepSeek 官方 API `https://api.deepseek.com`，`model=deepseek-v4-pro` |
| Qwen3-8B 训练后 | TRAIN 上 vLLM serve best checkpoint |

---

## §2 训练数据构建

### 2.1 核心原则

ToMBench 全集 2860 题、20 个 JSONL 文件**全部 hold-out 作为评测**。训练数据只能来自外部源 + 合成数据，与 ToMBench 严格零重叠。

### 2.2 五个数据源（合计 ~8k 训练 prompts）

| 源 | 数量 | 内容 | 获取方式 |
|---|---|---|---|
| Hi-ToM | ~2k | 多角色信念追踪、ToM order 0–4 location-belief | clone `bigai-ai/ToM-RL`，跑 `create_world.py / generate_tasks.py / generate_prompts.py` |
| ExploreToM | ~2k | Meta 的 A* 生成对抗性 ToM 故事 + MCQ | HuggingFace `facebookresearch/ExploreToM` |
| SimpleToM | ~1k | 应用场景 ToM（mental state + 行为预测） | HuggingFace 公开数据集 |
| SocialIQa | ~1.5k | 社会常识 + ToM 多选 | HuggingFace `allenai/social_i_qa` |
| deepseek-v4-pro 合成 | ~1.5k | 按 ToMBench 8 类**任务类型**生成新题（不复制具体题目） | 调 deepseek-v4-pro API |

### 2.3 统一训练格式（JSONL schema）

```json
{
  "messages": [
    {"role": "system", "content": "<ToM MCQ system prompt, see §4>"},
    {"role": "user", "content": "<Story>\n\nQuestion: <Q>\nA. <opt-A>\nB. <opt-B>\nC. <opt-C>\nD. <opt-D>"}
  ],
  "ground_truth": "A",
  "tag": "tom_mcq",
  "source": "hi_tom | exploretom | simpletom | socialiqa | synth"
}
```

### 2.4 合成数据防泄漏（关键）

1. **生成时禁止性 prompt**：明确告诉 deepseek "do NOT reproduce, paraphrase or translate any question from ToMBench by Chen et al."
2. **生成后过滤**：对每条合成题，计算 question + options 字符串 vs ToMBench 全集 2860 题的 MinHash + Jaccard 相似度（4-gram），阈值 0.6 以上整条丢弃
3. **内部去重**：合成数据内部也做 MinHash 去重
4. **人工抽样**：随机抽 50 条人工目视审查

### 2.5 数据切分

- **训练集**：8k 全部参与 RL 训练
- **阶段 1 子集**：随机 4k → `tom_train_4k.jsonl`
- **ToMBench 全集**：2860 题，评测专用
- **训练中评测子集**：随机 500 题 → `tombench_eval_subset500.jsonl`（每 50 步 ROLL 内置评测用）

### 2.6 数据混合策略

单一域，所有源 tag 统一为 `tom_mcq`，走同一个 `multiple_choice_boxed_rule_reward_worker`：

```yaml
domain_interleave_probs:
  tom_mcq: 1.0
```

### 2.7 数据构建脚本（`scripts/data/`）

- `build_hitom.py` / `build_exploretom.py` / `build_simpletom.py` / `build_socialiqa.py`
- `synth_tomtype.py`
- `merge_and_dedupe.py`
- `build_tombench_eval.py`

### 2.8 阶段 1 数据产物

- `data/tom/tom_train.jsonl` (~8k)
- `data/tom/tom_train_4k.jsonl` (4k 阶段 1 子集)
- `data/tom/tombench_eval.jsonl` (2860)
- `data/tom/tombench_eval_subset500.jsonl` (500)
- `data/tom/dedup_report.json`（防泄漏审计：合成数据与 ToMBench 的最高 Jaccard 相似度分布）

### 2.9 Go/No-go 判据

- 训练数据 ≥ 7000 条（允许 ±20% 浮动）
- `dedup_report.json` 中 max Jaccard ≤ 0.6
- 人工抽样 50 条无明显抄袭

---

## §3 评测协议与 Baseline 测量

### 3.1 评测脚本组织（`scripts/eval/`）

- `run_tombench.py` — 通用 ToMBench 评测器
- `clients.py` — 统一 OpenAI 兼容客户端（dashscope / deepseek / openai / local-vllm）
- `extractors.py` — 答案抽取器（direct boxed / CoT last boxed / DEL-ToM voting）
- `report.py` — 结果汇总成 markdown + json

### 3.2 协议 1 — Direct answer（主分数）

**System prompt**：
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

**抽取**：正则 `\boxed{([A-D])}` 取首个匹配；fallback 到响应中第一个大写字母 A-D。

**采样**：`temperature=0.0, top_p=1.0, max_tokens=32`

### 3.3 协议 2 — CoT（参考分数）

**System prompt**：
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Think step by step about the mental states of the characters,
then output your final answer in the format \boxed{X} where X is one of A, B, C, D.
Put your final \boxed{X} on the last line.
```

**抽取**：正则 `\boxed{([A-D])}` 取最后一个匹配；fallback 到响应末尾 200 字符内最后一个大写字母 A-D。

**采样**：`temperature=0.6, top_p=0.9, max_tokens=1024`

### 3.4 协议 3 — DEL-ToM 增强（可选）

仅对 belief 类子任务（False Belief, Unexpected Outcome, Knowledge-Attention Links 等）启用。每题生成 N=8 个 CoT 采样，多数投票。

**采样**：`temperature=0.7, top_p=0.95, n=8, max_tokens=1024`

仅在最终评测期使用，**训练期不用**。

### 3.5 Baseline 测量计划（阶段 2）

| 模型 | 接入方式 | 评测协议 |
|---|---|---|
| Qwen3-8B (non-thinking) | DashScope `https://dashscope.aliyuncs.com/compatible-mode/v1`，`qwen3-8b`，`extra_body: {"enable_thinking": false}` | direct + cot |
| Qwen3-8B (thinking) | 同上，`enable_thinking: true` | direct + cot |
| deepseek-v4-pro | `https://api.deepseek.com`，`deepseek-v4-pro` | direct + cot |

**每个模型 × 每个协议都对中文题和英文题各跑一遍**。ToMBench 同一题含中英两套字段（`故事/STORY`、`问题/QUESTION`、`选项A-D/OPTION-A-D`），中文题用中文 user prompt 模板，英文题用英文模板，独立计分。每模型每协议共 `2860 × 2 = 5720` 次 API 调用。

API 调用规范：
- 环境变量：`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`
- 并发：8 起步
- 重试：3 次指数退避
- 超时：60s
- 缓存：每次响应缓存到 `output/eval_cache/<model>_<protocol>_<question_id>.json`

### 3.6 评测报告产物

`output/eval/baseline_report.md` + `baseline.json`：

| Model | EN-direct | EN-cot | ZH-direct | ZH-cot | Overall-direct |
|---|---|---|---|---|---|
| Qwen3-8B (nt) | … | … | … | … | **Y_base_nt** |
| Qwen3-8B (t) | … | … | … | … | Y_base_t |
| deepseek-v4-pro | … | … | … | … | **X** |

+ 八子任务分项表。

训练目标按 **Y'_direct ≥ X_direct − 0.02** 衡量。

### 3.7 训练中评测

- 每 `eval_steps=50` 在 `tombench_eval_subset500.jsonl` 上跑 direct 协议
- 仅评 Qwen3-8B 当前 checkpoint（无 API 成本）
- 用 ROLL 内置 validation hook

### 3.8 Go/No-go 判据

- 三个 baseline 分数都成功落盘
- `Y_base_nt` 在 [0.50, 0.80] 区间（防 DashScope 模型 ID 错）
- 如 `X < Y_base_nt`：停下检查协议、prompt、抽取规则

---

## §4 System Prompt 与训练/评测模板

### 4.1 核心原则

**训练时的 prompt 模板 = 评测协议 1（Direct）的 prompt 模板**。训练分布与评测分布一致。

### 4.2 训练 system prompt（= 评测协议 1 system prompt）

```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

### 4.3 训练 user prompt 模板

**中文**：
```
故事：
{story}

问题：{question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
```

**英文**：
```
Story:
{STORY}

Question: {QUESTION}
A. {OPTION-A}
B. {OPTION-B}
C. {OPTION-C}
D. {OPTION-D}
```

合成数据按生成时的语种走对应模板。

### 4.4 设计取舍说明

| 选择 | 理由 |
|---|---|
| `\boxed{X}` 而非 `<answer>X</answer>` | 与 ROLL 现成 `MultipleChoiceBoxedRuleRewardWorker` 解析规则对齐 |
| 禁止解释性文字 | Qwen3-8B 在 ToMBench 上 thinking 模式不一定更优；抑制长 CoT 与 §5 R_len 配合 |
| 不加 `<think></think>` 双标签 | Social-R1 用双标签但其 R_len 仍是核心贡献项；简化结构降低噪声 |
| 中英各用母语 prompt | ToMBench 双语，避免强制翻译 |
| 统一 system prompt 不按源区分 | 避免模型学到 "source→格式" 捷径 |

### 4.5 chat template 组装（走 ROLL `template: qwen3`）

```
<|im_start|>system
<system prompt><|im_end|>
<|im_start|>user
<user prompt><|im_end|>
<|im_start|>assistant
```

assistant 段由模型生成，训练监督只计算 assistant 段的 token-level loss。

### 4.6 长度规范

- 训练期 `response_length: 256`（与 §5 R_len 的 L_max 对齐）
- 训练期 `prompt_length: 2048`
- 超长 prompt 处理：训练期 sample 级丢弃；评测期 story 截断到 1900 tokens

### 4.7 ground_truth 字段生成

- 所有源统一抽取为字母 `"A" | "B" | "C" | "D"`
- Hi-ToM / ExploreToM 若原非 MCQ，构建脚本里转 4 选 MCQ：正确答案 + 3 个启发式干扰项（最常见情况、其他角色 belief、其他 location）
- 合成数据由 deepseek 同时生成 4 选 + gold

### 4.8 显式不引入的复杂结构（防止后续摇摆）

- 不加 `<thinking>/<answer>` 双标签
- 不加 SIP 四阶段
- 不加 Social-R1 复杂长度门控以外的 reward 项（L2 仅保留 R_len 简化版）

---

## §5 Reward 设计（L2 主线 + L3 兜底）

### 5.1 L2 主线 reward 公式

```
R = R_fmt × R_out × R_len
```

| 分量 | 取值 | 计算规则 |
|---|---|---|
| `R_fmt` | {0, 1} | 响应中存在 `\boxed{X}` 且 X∈{A,B,C,D} 则 1 |
| `R_out` | {0, 1} | 抽取字母 == ground_truth 则 1 |
| `R_len` | (0, 1] | 长度窗口 sigmoid 门控 |

**乘性结构**：格式错或答错任何一项都使 R=0；只有格式对 + 答对才拿分，R_len 做精细调节。

### 5.2 R_len 长度窗口

```
R_len(L) = σ(k · (L − L_min) / (L_max − L_min)) × (1 − σ(k · (L − L_max) / (L_max − L_min)))
```

| 参数 | 值 | 说明 |
|---|---|---|
| L | int | 响应 token 数 |
| L_min | 8 | 输出 `\boxed{X}` + 换行的最小长度 |
| L_max | 256 | 短答案窗口上限 |
| k | 50 | sigmoid 陡峭度 |

**直觉行为**：
- L < 8：趋近 0
- 8 ≤ L ≤ 256：接近 1
- L > 256：快速衰减
- L >> 256：≈ 0（答对也几乎无分）

### 5.3 Reward worker 实现

新建 `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py`，基于 `multiple_choice_boxed_rule_reward_worker.py` 改写，新增 R_len。核心 < 80 行。

伪代码：

```python
class TomMcqRewardWorker(Worker):
    def __init__(self, worker_config):
        super().__init__(worker_config)
        self.l_min = worker_config.get("l_min", 8)
        self.l_max = worker_config.get("l_max", 256)
        self.k = worker_config.get("k", 50)

    def compute_rewards(self, data):
        for resp_tokens, gold in zip(...):
            response_text = decode(resp_tokens)
            extracted, fmt_ok = extract_boxed_letter(response_text)
            r_fmt = 1.0 if fmt_ok else 0.0
            r_out = 1.0 if extracted == gold else 0.0
            L = response_token_count(resp_tokens)
            r_len = sigmoid_window(L, self.l_min, self.l_max, self.k)
            r_total = r_fmt * r_out * r_len
            scores.append(r_total)
        return DataProto(scores=...)
```

### 5.4 Reward worker 配置

```yaml
rewards:
  tom_mcq:
    worker_cls: roll.pipeline.rlvr.rewards.tom_mcq_reward_worker.TomMcqRewardWorker
    tag_included: [tom_mcq]
    model_args:
      model_name_or_path: ${reward_pretrain}  # 仅复用 tokenizer
    data_args:
      template: qwen3
    world_size: 8
    infer_batch_size: 16
    l_min: 8
    l_max: 256
    k: 50
```

### 5.5 Reward logging 字段

```json
{
  "r_fmt": 0/1, "r_out": 0/1, "r_len": [0,1], "r_total": [0,1],
  "response_length": int, "extracted_letter": "A-D or null",
  "ground_truth": "A-D", "source_tag": "hi_tom/exploretom/..."
}
```

### 5.6 训练初期 reward 健康度检测（阶段 4 用）

| Step | 指标 | 目标 | 不达标处理 |
|---|---|---|---|
| 50 | `mean(R_fmt)` | > 0.95 | < 0.8 → system prompt 太弱 |
| 100 | `mean(R_out)` | > 0.8 × Y_base_nt | 显著低于 → 训练数据 bug |
| 150 | `mean(R_len)` | > 0.7 且非下降 | 持续下降 → 模型变啰嗦 |
| 200 | ToMBench subset500 direct | > Y_base_nt | 持平或下降 → 触发停训诊断 |

### 5.7 L3 兜底（仅 L2 未达 X−2pp 触发）

加入 Social-R1 的两个过程奖励项：

```
R = R_fmt × (w_o · R_out + τ(t) · (w_struct · R_struct + w_content · R_content)) × R_len
```

- `R_struct`：deepseek-v4-pro 作 judge，评 SIP 四阶段对齐度，0–1 浮点
- `R_content`：Qwen3-4B + LoRA pairwise RM，在 L2 训练中保存的 10 个 checkpoint 自采样得到的偏好对上训练（采样规则：K=6 trajectories × N=700 instances → ~42k segments → ~20k pairs）
- `τ(t)`：训练步数函数，前 30% τ=0（纯 outcome），后 70% 线性升到 1

L3 触发的额外 wall-clock：~50h（采样 6h + judge 评分 5h + RM 训练 24h + 第二阶段 RL 25h）。

---

## §6 训练超参与算法配置

### 6.1 算法

**GRPO 主算法** + 两项 DAPO 改进（ROLL 原生支持）：

- **Clip-Higher**：`use_pg_clip_range: true`，`pg_clip_low: 0.20`，`pg_clip_high: 0.28`
- **Dynamic Sampling**：`use_additional_prompts: true`，`max_running_requests: 256`

**不引入** DAPO 的 Token-Level Loss、Overlong Reward Shaping、GSPO（Qwen3-8B 是 dense，GSPO 主要利好 MoE，且 ROLL 未原生支持）。

### 6.2 主配置 `configs/tombench-rlvr/rlvr_config_stage1.yaml`

```yaml
exp_name: "qwen3-8B-tombench-rlvr"
seed: 42

max_steps: 200       # 阶段 1
save_steps: 50
logging_steps: 1
eval_steps: 50
resume_from_checkpoint: false

rollout_batch_size: 64
num_return_sequences_in_group: 8
ppo_epochs: 1
adv_estimator: "grpo"

prompt_length: 2048
response_length: 256

use_pg_clip_range: true
pg_clip_low: 0.20
pg_clip_high: 0.28
dual_clip_loss: true

reward_clip: 5
advantage_clip: 2.0
whiten_advantages: true
add_token_level_kl: false

max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95

use_additional_prompts: true
max_running_requests: 256
is_num_return_sequences_expand: false

pretrain: Qwen/Qwen3-8B
reward_pretrain: Qwen/Qwen3-8B

validation:
  data_args:
    template: qwen3
    file_name:
      - data/tom/tombench_eval_subset500.jsonl
  generating_args:
    max_new_tokens: 64
    top_p: 1.0
    top_k: -1
    temperature: 0.0
    num_return_sequences: 1
```

### 6.3 Actor 训练

```yaml
actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
    warmup_steps: 20
    num_train_epochs: 50  # 实际由 max_steps 控制
  data_args:
    template: qwen3
    file_name:
      - data/tom/tom_train.jsonl  # 阶段 1 改为 tom_train_4k.jsonl
    domain_interleave_probs:
      tom_mcq: 1.0
    dataset_dir: data/tom
    messages: messages
    interleave_probs: "1.0"
    preprocessing_num_workers: 16
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,16))
  infer_batch_size: 4
```

### 6.4 Rollout 推理（vLLM）

```yaml
actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.95
    top_k: 50
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      max_model_len: 4096
  device_mapping: list(range(0,12))
  infer_batch_size: 1
```

### 6.5 Reference 模型

```yaml
reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  data_args:
    template: qwen3
  strategy_args:
    strategy_name: megatron_infer
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
  device_mapping: list(range(0,16))
  infer_batch_size: 4
```

### 6.6 卡分配（16×H800 colocate）

| 角色 | 卡范围 | 用途 |
|---|---|---|
| actor_train | 0–15 | Megatron 训练 |
| actor_infer | 0–11 | vLLM rollout（与 train 错峰 colocate） |
| reference | 0–15 | KL 计算（与 train colocate） |
| rewards.tom_mcq | 0–7 | 纯规则，CPU 任务 |

ROLL offload manager 负责错峰加载/卸载权重。

### 6.7 阶段 1 → 阶段 2 升级

阶段 1 跑完 200 步且所有 §5.6 健康度检查通过后，升级到阶段 2：

```yaml
max_steps: 500
save_steps: 100
# 训练数据从 tom_train_4k.jsonl 切到 tom_train.jsonl
```

阶段 2 **不从阶段 1 ckpt 续训**，而是从 Qwen3-8B base 重新开始（避免 4k 子集的分布偏差污染）。

### 6.8 不调的超参（明确）

- `learning_rate: 1.0e-6`
- `gradient_accumulation_steps: 32`
- `warmup_steps: 20`
- `init_kl_coef`：保留 ROLL 默认
- `weight_decay: 0`
- `dtype: bf16`
- 并行：`TP=PP=EP=1`

### 6.9 Wall-clock 预估

| 阶段 | 数据 | 步数 | 每步 | 总 wall-clock |
|---|---|---|---|---|
| 阶段 1 | 4k | 200 | ~3 min | ~10h |
| 阶段 2 | 8k | 500 | ~3 min | ~25h |

**总 ~35h ≈ 1.5 天**（不触发 L3）。

---

## §7 端到端执行流程（跨机协作 + Docker 交付 + 自动化）

### 7.1 物理拓扑

| 节点 | 配置 | 职责 |
|---|---|---|
| **DEV（个人电脑）** | macOS Darwin 25.4.0，无 GPU | 写代码、构建数据、合成数据（调 deepseek API）、本地评测（调 DashScope/DeepSeek API）、分析、调方案 |
| **TRAIN（远程 16×H800）** | Linux + 16 GPU | 训练、训练中评测、ckpt 持久化、训练后服务推理 |
| **传输** | SSH + rsync 直连 | DEV↔TRAIN 同步代码、数据、ckpt、日志 |

### 7.2 仓库结构

```
training/
├── framework/ROLL/                          # 已有 ROLL（仅新增一个 reward worker）
│   └── roll/pipeline/rlvr/rewards/
│       └── tom_mcq_reward_worker.py
├── docker/
│   ├── dev/{Dockerfile, docker-compose.yml}
│   ├── train/{Dockerfile, docker-compose.yml, entrypoint.sh}
│   └── serve/{Dockerfile, docker-compose.yml, entrypoint.sh}
├── scripts/
│   ├── data/                                # 数据构建（DEV）
│   ├── eval/                                # 评测（DEV，调 API 或远程 vLLM）
│   ├── deploy/                              # 跨机操作（DEV 触发）
│   │   ├── sync_to_train.sh / sync_from_train.sh
│   │   ├── train_launch.sh / serve_launch.sh
│   │   ├── train_monitor.py / track_best_ckpt.py
│   │   └── convert_megatron_to_hf.py
│   ├── analysis/
│   └── env_check.py
├── configs/
│   ├── tombench-rlvr/
│   │   ├── rlvr_config_stage1.yaml
│   │   ├── rlvr_config_stage2.yaml
│   │   └── rlvr_config_stage3_l3.yaml
│   ├── deploy.env.example                   # 入仓
│   └── deploy.env                           # gitignore
├── data/                                    # gitignore
├── output/                                  # gitignore
├── docs/superpowers/specs/                  # 本文档
└── Makefile
```

### 7.3 Docker 镜像（三个）

**`dev` 镜像**（DEV macOS 本地，无 GPU）：
- 基础：`python:3.10-slim`
- 安装：`openai`, `datasets`, `pyarrow`, `pandas`, `matplotlib`, `tensorboard`, `tqdm`, `datasketch`
- 用途：数据构建、合成、评测、分析
- 挂载：`./scripts`、`./data`、`./output`，环境变量 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY` 透传

**`train` 镜像**（TRAIN，CUDA）：
- 基础：`nvcr.io/nvidia/pytorch:24.10-py3`（CUDA 12.4）
- 安装：ROLL `requirements_torch280_vllm.txt` + `pip install -e framework/ROLL`
- 入口流程：等数据就绪 → 校验模型缓存 → 启 ray → 跑 `start_rlvr_pipeline.py` → 写 ckpt → exit
- 挂载：`/mnt/data`、`/mnt/models`、`/mnt/output`，`--gpus all`
- 镜像 ~25GB，multi-stage build 剥离构建期依赖

**`serve` 镜像**（TRAIN，训后启）：
- 基础：与 `train` 同（共享 vLLM 层）
- 入口：`vllm.entrypoints.openai.api_server --model /mnt/output/final_model --port 8000`
- 挂载：`/mnt/output/final_model`
- 端口：8000

### 7.4 Makefile 自动化入口

```makefile
# === DEV 本地 ===
build-data:
	docker compose -f docker/dev/docker-compose.yml run --rm dev \
	  python scripts/data/merge_and_dedupe.py

baseline:
	docker compose run --rm dev \
	  python scripts/eval/run_tombench.py --preset baseline-all

# === 部署到 TRAIN ===
sync-up:
	bash scripts/deploy/sync_to_train.sh

train-stage1:
	ssh $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/train/docker-compose.yml \
	  --env-file configs/deploy.env \
	  run --rm -e STAGE=stage1 train"

train-stage2:
	ssh $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose run --rm -e STAGE=stage2 train"

train-stage3-l3:
	ssh $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose run --rm -e STAGE=stage3_l3 train"

sync-down:
	bash scripts/deploy/sync_from_train.sh

# === 部署训练后模型 ===
serve-launch:
	ssh $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/serve/docker-compose.yml up -d serve"

serve-stop:
	ssh $(TRAIN_HOST) "cd $(TRAIN_PATH) && \
	  docker compose -f docker/serve/docker-compose.yml down"

# === DEV 评测训练后模型 ===
eval-final:
	docker compose run --rm dev \
	  python scripts/eval/run_tombench.py \
	    --backend openai \
	    --base-url http://$(TRAIN_HOST):8000/v1 \
	    --model qwen3-8b-tom \
	    --protocols direct,cot,del_tom \
	    --output output/eval/final.json

# === 分析 ===
analyze:
	docker compose run --rm dev \
	  python scripts/analysis/plot_training_curves.py && \
	  python scripts/analysis/error_audit.py && \
	  python scripts/analysis/diff_eval_results.py

# === 串联 ===
pipeline-stage1: build-data baseline sync-up train-stage1 sync-down analyze
pipeline-stage2: sync-up train-stage2 sync-down serve-launch eval-final analyze
pipeline-l3:     sync-up train-stage3-l3 sync-down serve-launch eval-final analyze
```

### 7.5 部署配置 `configs/deploy.env.example`

```bash
TRAIN_HOST=user@training-server.example.com
TRAIN_PATH=/data/cpfs_0/projects/qwen3-tom
TRAIN_SSH_KEY=~/.ssh/id_rsa_train
TRAIN_DATA_DIR=/data/cpfs_0/tom-data
TRAIN_MODELS_DIR=/data/cpfs_0/models
TRAIN_OUTPUT_DIR=/data/cpfs_0/tom-output

DEV_DATA_DIR=./data
DEV_OUTPUT_DIR=./output
```

API keys 不入文件，从 shell 透传到 docker compose。

### 7.6 同步脚本

**`sync_to_train.sh`**：rsync 代码 + 配置 + 数据（排除 `output`、`data`、`.git`、`__pycache__`、`*.pyc`）。

**`sync_from_train.sh`**：rsync best checkpoint + tensorboard logs + eval results。仅拉 best，不拉所有中间 ckpt。

### 7.7 阶段总览

| 阶段 | 在哪 | 命令 | 时间 |
|---|---|---|---|
| 0 环境检查 | DEV+TRAIN | `make env-check` | 0.5h |
| 1 数据构建 | DEV | `make build-data` | 4–6h |
| 2 baseline | DEV | `make baseline` | 0.5h |
| 3 reward worker | DEV | `make test-reward` | 2h |
| 4 训练阶段 1 | TRAIN | `make pipeline-stage1` | 10h |
| 5 训练阶段 2 | TRAIN | `make pipeline-stage2` | 25h |
| 6 最终评测 | DEV→远程 vLLM | (含于 stage2) | 0.5h |
| 7 分析 | DEV | `make analyze` | 即时 |
| 8 调方案重训 | DEV→TRAIN | 改配置 + `make pipeline-stage2` | 视改动 |
| 9 L3 兜底 | TRAIN | `make pipeline-l3` | +50h |

**主线总 wall-clock**：~45–50h（不触发 L3）；触发 L3 ~100h。

### 7.8 自动化调整闭环

每次评测后 DEV 上跑 `make analyze` 自动产出：

- `output/analysis/curves.png` — 训练曲线
- `output/analysis/eval_diff.md` — vs base / deepseek 差异表 + 子任务进退步
- `output/analysis/errors.md` — 训后仍答错样本按子任务抽样

人工 review 后调整动作：

| 观察 | 调整 | 改动位置 |
|---|---|---|
| 某子任务系统性错 | 增加该类型合成数据 | `synth_tomtype.py --task faux_pas --n 500` |
| reward 早期 saturate | 提升数据难度 | `merge_and_dedupe.py --difficulty_filter hard` |
| response_length 持续上升 | 收紧 L_max（256→128） | `rlvr_config_stage2.yaml` 一行 |
| 中文显著低于英文 | 提升中文比例 | `merge_and_dedupe.py --zh_ratio 0.5` |
| 未达 X−2pp | 启动 L3 | `make pipeline-l3` |

### 7.9 单元 / 集成测试

| 测试 | 在哪 | 内容 |
|---|---|---|
| `pytest scripts/data/tests/` | DEV docker | 数据 schema + MinHash |
| `pytest scripts/eval/tests/` | DEV docker | 答案抽取边界 |
| `pytest framework/ROLL/tests/test_tom_mcq_reward.py` | DEV docker | reward worker 6 单测 |
| 端到端 dry-run | DEV docker | 10 条 mini 数据 + Qwen2.5-0.5B 跑 1 步训练（CPU） |

---

## §8 风险、监控、回退

### 8.1 风险清单

| # | 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | 数据泄漏 | 中 | 致命 | MinHash 双向去重 + 阈值 0.6 + 人工抽样 |
| R2 | reward hacking（答案分布塌缩） | 中 | 高 | 乘性 reward + 监控答案分布熵 |
| R3 | 训练不稳定（KL 爆炸/NaN/熵塌缩） | 中 | 高 | Clip-Higher + dual_clip_loss + 自动早停 |
| R4 | 训测协议不一致 | 低 | 中 | 主分数 direct，训测同模板 |
| R5 | rsync 中断 | 中 | 低 | `--partial` + checksum retry |
| R6 | API rate limit/欠费 | 中 | 低 | 指数退避 + 缓存 |
| R7 | thinking/non-thinking 混淆 | 低 | 中 | 显式 `extra_body.enable_thinking` 标注 |
| R8 | Docker CUDA 版本冲突 | 中 | 中 | nvcr.io 基础镜像 + 固定版本 |
| R9 | ckpt 格式转换失败 | 中 | 高 | 阶段 1 就验证转换脚本 |
| R10 | deepseek-v4-pro API 变更 | 低 | 高 | 目标 X 落盘后固化 |
| R11 | 训练中 subset500 震荡 | 高 | 低 | best-checkpoint 跟踪 |
| R12 | SSH 断开 | 中 | 高 | docker compose detached 模式 |

### 8.2 训练监控指标

**ROLL 原生**：
- `critic/rewards/mean`、`critic/rewards/std`
- `actor/loss`、`actor/kl`、`actor/entropy`
- `actor/ppo_ratio_high_clipfrac`、`actor/ppo_ratio_low_clipfrac`
- `critic/advantages/mean`、`critic/advantages/std`
- `response_length/mean`、`response_length/max`

**L2 新增**：
- `reward/r_fmt_mean`：> 0.95
- `reward/r_out_mean`：从 Y_base_nt 单调上升
- `reward/r_len_mean`：> 0.7
- `reward/answer_distribution`：A/B/C/D 接近均匀（reward hacking 检测）

### 8.3 自动早停

| 条件 | 动作 |
|---|---|
| `actor/kl > 0.5` 连续 3 个 eval | 暂停、回退、降 lr 50% 继续 |
| `actor/entropy < 0.1` 连续 3 个 eval | 熵塌缩，停 |
| subset500 连续 3 次 < `best − 0.03` | 已过峰值，停 |
| `r_len_mean` 持续下降 > 0.1 | 告警（不强停） |
| 某字母占比 > 0.6 | reward hacking 告警 + subset500 eval |

**实现**：外挂 `scripts/deploy/train_monitor.py` 守护进程，60s 一次读 tb 日志，达条件 `docker kill <train_container>`；best ckpt 已周期性保存。

### 8.4 Checkpoint 与 best

- `save_steps: 100`，500 步训练共 5 个 ckpt
- 保留策略：`keep_last_n: 3` + always keep best
- `track_best_ckpt.py` 每 50 步从 tb 读 subset500 分数，symlink `best_checkpoint/` 指向最高分
- 转换：`scripts/convert_megatron_to_hf.py` 把 best 转 HF 格式到 `output/final_model/`（供 vLLM serve）

### 8.5 回退策略

| 失败场景 | 回退动作 | RTO |
|---|---|---|
| 数据源下载失败 | 白名单去掉，其他源补齐 | <1h |
| baseline API 失败 | 失败题缓存重试，3 轮失败标记后剔除 | <0.5h |
| 阶段 4 Go/No-go 不过 | 按调整表修 config 重跑 | 10h |
| 阶段 5 OOM/NCCL | `resume_from_checkpoint: true` 从最近 ckpt 恢复 | ≤50 步 |
| SSH 断开 | Docker detached 无影响 | 0 |
| 阶段 6 未达目标 | 触发 L3 | +50h |
| ckpt 转换失败 | fallback Megatron 原生推理 | <1h |

### 8.6 manifest 与可追溯

每阶段产 `<stage>_manifest.json`：产物列表+sha256、输入数据版本（git+sha256）、config snapshot、Docker image digest、起止时间、API 调用统计。

### 8.7 成本监控

- API 调用 → `output/api_usage.jsonl`，`make report-cost` 聚合
- GPU 时：`nvidia-smi --query-gpu=...` 后台记录
- 磁盘：sync-down 后 TRAIN 上清理非 best ckpt

### 8.8 已知局限

- 跨论文 baseline 不可对比（评测协议差异），只追内部相对提升
- deepseek-v4-pro X 待测才知
- ROLL upstream bug 可能阻塞
- 不做中英 reward 权重平衡

---

## §9 交付物清单

### 9.1 代码（入 git）

- `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py`
- `framework/ROLL/tests/test_tom_mcq_reward.py`
- `scripts/data/{build_*,synth_tomtype,merge_and_dedupe}.py`
- `scripts/eval/{run_tombench,clients,extractors,report}.py`
- `scripts/deploy/{sync_*.sh, train_launch.sh, serve_launch.sh, train_monitor.py, track_best_ckpt.py, convert_megatron_to_hf.py}`
- `scripts/analysis/{plot_training_curves,diff_eval_results,error_audit}.py`
- `scripts/env_check.py`
- `Makefile`

### 9.2 Docker

- `docker/{dev,train,serve}/{Dockerfile, docker-compose.yml, entrypoint.sh}`
- `docker/.dockerignore`

### 9.3 配置

- `configs/tombench-rlvr/rlvr_config_stage{1,2,3_l3}.yaml`
- `configs/deploy.env.example`（入仓）+ `configs/deploy.env`（gitignore）

### 9.4 文档

- `docs/superpowers/specs/2026-05-11-qwen3-8b-tombench-rlvr-design.md`（本文档）
- `docs/README.md`
- `docs/runbook.md`
- `docs/data-card.md`
- `docs/eval-protocol.md`

### 9.5 运行时产物（gitignore）

DEV：
- `data/tom/{tom_train,tom_train_4k,tombench_eval,tombench_eval_subset500}.jsonl`
- `data/tom/dedup_report.json`
- `output/eval/{baseline_report,final_report}.md` + `*.json`
- `output/analysis/{curves.png,eval_diff.md,errors.md}`
- `output/tensorboard/`
- `output/checkpoints/best/`（HF 格式）
- `output/api_usage.jsonl`

TRAIN：
- `$TRAIN_OUTPUT_DIR/{checkpoint-*, best_checkpoint, final_model, tensorboard}/`

### 9.6 manifest

`output/manifests/stage{0..7}_*.json`

### 9.7 最终交付包

`qwen3-8b-tombench-rlvr-v1.0.tar.gz`：
- `code/` 整个 repo 快照
- `data/tom_train.jsonl`、`tombench_eval.jsonl`
- `docker_images/{train,serve,dev}.tar`
- `models/final_model_hf/` + sha256
- `reports/{baseline,final}_report.md`、`analysis/*`、`data-card.md`
- `manifests/*`
- `README.md`（复现步骤）

### 9.8 不交付

- 训练中间 ckpt（仅 best）
- SocialPairs-20K（仅 L3 触发存在）
- TensorBoard 原始事件文件
- API 响应缓存
- Megatron 格式原始 ckpt

---

## 附录 A — 关键调研文献与数字

| 文献 | 关键结论 | 对设计影响 |
|---|---|---|
| ToMBench (ACL 2024, Chen et al., arxiv 2402) | 2860 题、8 任务、中英双语、MCQ、`\boxed{答案}` 字母格式 | 评测集结构、答案抽取规则 |
| ToM-RL (BIGAI, arxiv 2504.01698) | Qwen2.5 + GRPO + 纯规则 reward (+1 fmt / +2 ans) + Hi-ToM 训练，达 Hi-ToM 84.50% 超 GPT-4o | L2 简单 reward 路线、Hi-ToM 作训练数据源 |
| Social-R1 (Tsinghua, arxiv 2603.09249) | Qwen3-8B + GRPO + 5 因子 reward (R_fmt × (w_o·R_out + τ·(w_struct·R_struct + w_content·R_content)) × R_len)，ToMBench overall 0.6179 → 0.6881 | R_len 公式直接复用；R_struct/R_content 作 L3 兜底 |
| DAPO (ByteDance, arxiv 2503.14476) | Clip-Higher (0.2/0.28) + Dynamic Sampling + Token-Level Loss + Overlong Shaping，AIME 50% | 保留 Clip-Higher + Dynamic Sampling；不引入 Token-Level Loss（短答案无需） |
| Dr.GRPO (Sea AI Lab, arxiv 2503.20783, ICLR 2026) | 移除 1/\|o_i\| 和 std(R) 两个归一化偏置 | 暂不引入（与 DAPO Clip-Higher 同时上变量太多） |
| GSPO (Qwen team, arxiv 2507.18071) | 序列级 importance ratio，Qwen3 系列训练用 | 不引入（dense 模型收益小，ROLL 未原生支持） |
| DEL-ToM (EMNLP 2025) | inference-time scaling，belief tracking 分解 + 验证器 | 作为可选协议 3，仅评测期使用 |
| "To Think or Not To Think" (arxiv 2602.10625) | 报告 Qwen3-8B nt 0.729 / t 0.680 | 仅作预期参照，不作 baseline |

## 附录 B — DeepSeek API 调用规范

- Base URL：`https://api.deepseek.com`
- Model ID：`deepseek-v4-pro`
- OpenAI 兼容：是
- Context：1M tokens
- 环境变量：`DEEPSEEK_API_KEY`
- 重试：3 次指数退避
- 超时：60s

## 附录 C — DashScope API 调用规范

- Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- Model ID：`qwen3-8b`
- OpenAI 兼容：是
- thinking 切换：`extra_body: {"enable_thinking": bool}`
- 环境变量：`DASHSCOPE_API_KEY`

---

**文档结束**
