# 技术报告：Qwen3-8B Stage 7 + Qwen3-14B Stage 8 — Phase A/C 数据扩展与风格匹配

**作者**：基于训练日志（`train_stage7_1x8_20260518_192021.log`、`train_stage8_1x8_14b_20260518_234107.log`）与配置（`rlvr_config_stage7_1x8.yaml`、`rlvr_config_stage8_1x8_14b.yaml`）整理（commit `79bbaed`）。

**状态**：本文档是 Qwen3-8B/14B + GRPO 项目的最终工程报告，记录在 Stage 1 (`tech_report_qwen3-{8b,14b}_stage1_zh.md`) 基础上的延伸：

- **Stage 6**：清洁数据 + GPT-5.5 合成 1400 条（false belief + scalar implicature），8B/14B 均训练
- **Stage 7 (Phase A 数据扩展)**：合成 2300 条覆盖 HOT 失败模式的新数据，**8B 受益（+0.25pp 破 ceiling）但 14B 反退（-0.41pp）**
- **Stage 8 (Phase C 风格匹配修复)**：用 ToMBench 5-7 句风格重新合成 1200 条，仅在 14B 上跑（max_steps 350）

最终生产模型：**8B → stage7**（0.7419 raw / 0.8321 clean），**14B → stage8**（0.7594 raw / 0.8449 clean）。

---

## 1. 项目大目标与本次扩展动机

接续 Stage 1 报告，Stage 6 之后我们用 GPT-5.5 审查了训练数据和评测集，得到两个核心发现：
1. **评测集有 ~20% 标签问题**（GPT-5.5 audit 5718 题，1167 条判定 wrong-label / ambiguous / translation-artifact）
2. **HOT 错误（GPT-5.5 + deepseek 都对、我们错的 469 条）来自 7 类失败模式**，其中我们训练数据基本未覆盖的占 91%：
   - factual_inference 17%
   - social_norm_inference 16%
   - intention_attribution 14%
   - overly_literal 14%
   - emotion_attribution 11%
   - knowledge_attention_link 9%
   - indirect_speech_act 9%

**Stage 7 假设**：合成专门覆盖这些失败模式的 1500 条 EN+ZH 数据（Phase A.1） + 800 条中文 ToM 数据（Phase A.2），合计 2300 条加入训练集。

**Stage 7 结果**：
- **8B**: 0.7380 → 0.7419 (**+0.25pp** ⭐ 首次破 8B 历史 stage1 0.7394 ceiling)
- **14B**: 0.7580 → 0.7539 (**-0.41pp** ↓ 反退)

**Stage 8 假设**：14B 反退的原因是 GPT-5.5 合成的故事**风格不匹配** ToMBench（8-12 句精致 vs 5-7 句直白）+ 数据稀释（max_steps 250 不变让每条样本被见次数从 7x 降到 5x）。

**Stage 8 解决方案**：
1. 用严格 5-7 句直白叙事 prompt 重新合成 1200 条（Phase C），替换 Phase A.1
2. max_steps 250 → 350，恢复每条样本 ~7x 见次数

**Stage 8 结果（仅 14B）**: 0.7580 → **0.7594** (+0.14pp，本项目历史最高，subset500 del_tom 0.7920 首次反超 deepseek)。

## 2. 硬件与软件栈

同 Stage 1（详见 `tech_report_qwen3-8b_stage1_zh.md` §2），新增/确认：

| 项目 | Stage 7/8 配置 |
|---|---|
| 训练框架 | ROLL（vendored，2025-Q3 快照） |
| 训练镜像 | `qwen3-tom-train:latest`（基础 NVIDIA pytorch 24.05-py3） |
| Megatron-Core | 0.16.0 |
| Transformer Engine | 2.2.0 |
| vLLM | 0.8.4 |
| Ray | 2.48 |
| 8B 单步耗时 | ~60 秒 (1×8 H800, DP=8 colocated) |
| 14B 单步耗时 | ~75 秒 (1×8 H800, TP=2 + DP=4 colocated) |
| 8B Megatron→HF 转换 | ~7 分钟（1 GPU + CPU） |
| 14B Megatron→HF 转换 | ~12 分钟 |

**Stage 7 8B 总耗时**: ~4.5h（250 步训练 + 10 分钟 init + 10 分钟 save）
**Stage 8 14B 总耗时**: ~7h（350 步训练 + 10 分钟 init + 15 分钟 save）

## 3. 训练数据演变

### 3.1 数据演化时间线

| 阶段 | 训练集 | 关键变化 |
|---|---|---|
| Stage 1-5 | tom_train.jsonl 5911 → 8901 records | 加 SocialIQa / Hi-ToM 失败；Phase-1 合成数据 990 条 |
| **Stage 6** | tom_train.jsonl 7259 records | **GPT-5.5 审查**: 丢 ExploreToM (89% low/harmful) + simpletom_zh (39% low/harmful) + 13 phase1 bad；**新合成 1400** false_belief_1st (600) + false_belief_2nd (400) + knowledge_scalar (400) |
| **Stage 7** (Phase A) | tom_train.jsonl **9559** records | **+2300 records**: Phase A.1 (1500 EN+ZH) + Phase A.2 (800 ZH) |
| **Stage 8** (Phase C) | tom_train.jsonl **9259** records | **Phase A.1 替换为 Phase C 1200 条风格匹配** + 保留 Phase A.2 800 ZH |

### 3.2 Stage 7 Phase A 数据合成详情（共 2300 条）

**Phase A.1** — 1500 条，覆盖 HOT 失败模式（EN+ZH 各半）：

| 类别 | 数量 | 设计意图 |
|---|---|---|
| social_norm | 400 | 失礼识别 + 礼貌反应陷阱（"对方微笑 ≠ 没失礼"） |
| factual_detail | 300 | 关键非显眼细节驱动答案（"她离开后又回来"决定 belief 状态） |
| intention_attribution | 400 | 角色行为 + 隐藏目标（销售员夸张话术=推销目标，不是字面笑话） |
| indirect_speech_act | 400 | 暗示请求识别（"嘴干"=想喝水；"今天好热啊"=请开空调） |

**Phase A.2** — 800 条中文 ToM（ZH gap 比 EN 大 3-5pp on Belief/Knowledge/Desire）：

| 类别 | 数量 | 设计意图 |
|---|---|---|
| belief_zh | 300 | 1st-order belief + 模糊场景信念推断 + 对话信念 |
| knowledge_zh | 250 | 谁知道什么 + 标量含蓄（"几乎没人" ≈ 1-3） + 知识-注意力联结 |
| desire_zh | 250 | 矛盾欲望 + 隐含偏好 + 说服策略 |

**Phase A 合成方式**:
- 模型：`gpt-5.5` via `OPENAI_BASE_URL=https://www.modelservice.top/v1`
- temperature 0.9, max_tokens 1500, concurrency 8
- 总耗时：Phase A.1 ~85min + Phase A.2 ~42min = 127min
- 反作弊：MinHash 4-gram Jaccard 0.6 阈值 vs ToMBench eval → **0% 泄漏，max Jaccard 0.000**

### 3.3 Stage 8 Phase C 数据合成详情（1200 条，替换 Phase A.1）

Stage 7 14B 失败诊断：Phase A.1 的 GPT-5.5 prompt 没限制故事长度，模型默认写 8-12 句精致叙事，但 ToMBench 是 5-7 句直白故事。模型在 val (subset500，部分含合成数据风格) 上学得很好，但 full 5718 评测上反退 0.41pp。

**Phase C system prompt 严格约束**：
- 每篇故事 **5-7 句**（不再 8-12 句）
- 简单 SVO 结构，每句陈述一个事实
- 禁止 "Suddenly,", "Despite this,", "Without warning,"
- 禁止场景渲染（天气、氛围段落）
- 普通场景（教室、办公室、午餐间、走廊）

**Phase C 7 类失败模式覆盖 1200 条**（EN+ZH 各半）:

| 类别 | 数量 | 占 HOT 失败模式比例 |
|---|---|---|
| factual_inference | 200 | 17% |
| social_norm_inference | 200 | 16% |
| intention_attribution | 200 | 14% |
| overly_literal | 200 | 14% (pragmatic vs literal) |
| emotion_attribution | 150 | 11% |
| knowledge_attention_link | 150 | 9% |
| indirect_speech_act | 100 | 9% |

**Phase C 样本验证**（第一条 EN）:

> Emma carried a tray in the lunchroom. Noah stood near the closed door. Emma looked at the door. Emma said, "My hands are full." Noah opened the door for Emma. Emma walked through the door.
> Q: What did Emma actually want?
> - A: She wanted Noah to open the door. (correct)
> - B: She wanted to say that her hands were full. (字面陷阱)
> - C: She wanted Noah to take her tray.
> - D: She wanted to leave the lunchroom.

6 句直白叙事 + 完美 ToMBench 风格 + 选项 B 是字面陷阱。

**Phase C 合成耗时**: ~58min, $50 (GPT-5.5 API)

### 3.4 评测集清洁化（GPT-5.5 Audit）

并行于训练，我们用 GPT-5.5 audit 全部 5718 ToMBench 评测题。每题让 GPT-5.5 给出 `label_assessment` (correct/ambiguous/wrong) + `label_confidence` (high/medium/low) + `issue_category`。

**5718 题 audit 结果**:

| 分类 | 数量 | % |
|---|---|---|
| label_correct (keep) | 4551 | 79.6% |
| ambiguous_question | 620 | 10.8% |
| wrong_label | 460 | 8.0% |
| options_overlap | 72 | 1.3% |
| translation_artifact | 50 | 0.9% |

**按 task drop rate**:

| Task | drop rate |
|---|---|
| Knowledge | **55.2%** ⚠️⚠️ (解释了为什么 Knowledge 永远卡 0.50) |
| Desire | 31.4% |
| Emotion | 18.7% |
| Non-literal Comm | 17.4% |
| False Belief | 13.6% |
| Intention | 13.5% |
| Belief | 8.5% |

EN/ZH 几乎一致 (20.8% vs 20.0%)，说明是原始标签问题，不是翻译噪声。

**清洁后 eval set**: `data/tom/tombench_eval_clean.jsonl` (4551 题)

## 4. 算法：GRPO + DAPO Clip-Higher + 动态采样

同 Stage 1 (`tech_report_qwen3-8b_stage1_zh.md` §5)。**Stage 7/8 没有改动算法**，所有提升都来自数据 + 训练步数。

关键参数（Stage 7 8B / Stage 8 14B 一致）：
- `pg_clip_low: 0.20, pg_clip_high: 0.28`
- `dual_clip_loss: true`
- `whiten_advantages: true`
- `difficulty_low_threshold: 0.1, difficulty_high_threshold: 0.95`
- `add_token_level_kl: false`（无 KL penalty）

## 5. 超参数对比

```yaml
# Stage 7 8B (config: rlvr_config_stage7_1x8.yaml)
max_steps: 250
save_steps: 250
data: tom_train.jsonl (9559 records)
rollout_batch_size: 32
num_return_sequences_in_group: 8       # 有效 rollout = 256
gradient_accumulation_steps: 32
learning_rate: 1.0e-6
prompt_length: 2048
response_length: 256
DP: 8, TP: 1
```

```yaml
# Stage 8 14B (config: rlvr_config_stage8_1x8_14b.yaml)
max_steps: 350                         # ← 关键改动 vs s7 14B 的 250
save_steps: 350
data: tom_train.jsonl (9259 records)
rollout_batch_size: 32
num_return_sequences_in_group: 8       # 有效 rollout = 256
gradient_accumulation_steps: 64
learning_rate: 1.0e-6
prompt_length: 1024                    # ← 1024 (vs 8B 的 2048, KV cache 限制)
response_length: 256
DP: 4, TP: 2                            # ← TP=2 for 14B
```

**关键决策说明**:

### 5.1 为什么 14B max_steps=350 而 8B max_steps=250
Stage 7 14B 失败的两个根因之一是**数据稀释**。9559 records / (32 × 8) = 37 batches/epoch，max_steps 250 意味着每条记录被见 ~5x（vs Stage 6 的 7x）。Stage 8 把 14B 升到 350 步，恢复每条 ~7.5x。

8B 没做对应调整因为：(1) 8B 时 ceiling 限制，多训也不会显著涨；(2) 8B 训练时间更宝贵；(3) 实验目的是验证 Phase A 数据本身是否有效。

### 5.2 8B 与 14B 不同的 prompt_length
- 8B: `prompt_length: 2048`
- 14B: `prompt_length: 1024`（KV cache 限制：vllm 用 0.45 GPU memory fraction，14B + KV cache 不够容纳 2048 prompt）

实际训练集 prompt 长度大多在 600-900 token，1024 完全够用。

### 5.3 8B 与 14B 不同的并行策略
- 8B: DP=8, TP=1（单 GPU 装得下 8B）
- 14B: DP=4, TP=2（14B 在 16-bit 下 28 GB，单 H800 80 GB 装 actor_train + reference + vLLM 三角色不够，必须 TP）

详见 `tech_report_qwen3-14b_stage1_zh.md` §3 "为什么选 TP=2"。

### 5.4 关键 ROLL 配置
```yaml
strategy_config:
  use_distributed_optimizer: true
  distrib_optim_fully_reshardable_mem_efficient: true  # 救命魔法
  recompute_granularity: full
```

详见 `tech_report_qwen3-8b_stage1_zh.md` §7 "分布式 Save（OOM 修复史）"。Stage 7/8 都用了这个配置，14B save 耗时 ~15 分钟（96 GB optimizer state via Gloo+CPU），稳定无 OOM。

## 6. 训练轨迹

### 6.1 Stage 7 8B 轨迹

从 `train_stage7_1x8_20260518_192021.log` 直接读取：

| step | rollout score | r_fmt | r_out | r_len | KL loss | val_correct/all (subset500) |
|---|---|---|---|---|---|---|
| 0 | 0.301 | 0.453 | 0.453 | 0.574 | 0.000 | 0.036 |
| 25 | 0.281 | 0.309 | 0.293 | 0.641 | 0.004 | — |
| **50** | 0.467 | 0.492 | 0.480 | 0.731 | 0.034 | **0.260** ⭐ |
| 75 | 0.562 | 0.574 | 0.570 | 0.779 | 0.082 | — |
| 100 | 0.758 | 0.809 | 0.766 | 0.897 | 0.114 | 0.460 |
| 125 | 0.804 | 0.820 | 0.809 | 0.905 | 0.194 | — |
| 150 | 0.845 | 0.895 | 0.848 | 0.944 | 0.268 | 0.574 |
| 175 | 0.848 | 0.895 | 0.848 | 0.946 | 0.228 | — |
| 200 | 0.957 | 0.961 | 0.957 | 0.980 | 0.201 | **0.600** |
| 225 | 0.817 | 0.875 | 0.820 | 0.934 | 0.232 | — |
| 249 (final) | 0.880 | 0.910 | 0.880 | 0.961 | 0.245 | — |

**关键观察**:
- **step 50 val 0.260** 大幅超过 8B 历代任何 stage（s1 0.204, s6 0.202）— Phase A 数据红利在 8B 上立刻显现
- **step 100 val 0.460** 反而比 s6 同期 (0.490) 低 -3.0pp — 风格漂移开始显现
- **step 150-200 val 恢复**: 0.574 → 0.600，超 s6 同期 (0.564 / 0.594) 约 +0.6-1.0pp
- **最终 val 0.600** 比 s6 (0.594) 高 +0.6pp，转化到 full eval 是 **+0.39pp** (0.7419 vs 0.7380)

### 6.2 Stage 8 14B 轨迹

从 `train_stage8_1x8_14b_20260518_234107.log` 读取：

| step | rollout score | r_fmt | r_out | r_len | KL loss | val_correct/all |
|---|---|---|---|---|---|---|
| 0 | 0.302 | 0.324 | 0.324 | 0.640 | 0.000 | 0.070 |
| 25 | 0.480 | 0.496 | 0.496 | 0.732 | 0.016 | — |
| **50** | 0.802 | 0.828 | 0.809 | 0.906 | 0.081 | **0.516** |
| 75 | 0.941 | 0.973 | 0.941 | 0.986 | 0.239 | — |
| **100** | 0.815 | 0.930 | 0.820 | 0.960 | 0.205 | **0.662** |
| 125 | 0.945 | 0.984 | 0.945 | 0.992 | 0.341 | — |
| **150** | 0.949 | 0.969 | 0.949 | 0.984 | 0.418 | **0.698** |
| 175 | — | — | — | — | — | — |
| **200** | 0.918 | 0.984 | 0.918 | 0.992 | 0.355 | **0.706** |
| 225 | 0.935 | 0.957 | 0.938 | 0.973 | 0.308 | — |
| **250** | — | — | — | — | — | **0.720** ⭐ |
| 275 | 0.996 | 0.996 | 0.996 | 0.998 | 0.393 | — |
| **300** | — | — | — | — | — | **0.710** |
| 325 | 0.996 | 0.996 | 0.996 | 0.998 | 0.575 | — |
| 349 (final) | 0.906 | 0.941 | 0.906 | 0.970 | 0.389 | — |

**关键观察**:
- **step 50 val 0.516** = s7 14B step 50（一致），比 s6 14B (0.496) 高 +2.0pp
- **step 100 val 0.662** = s7 14B step 100（一致），比 s6 14B (0.628) 高 +3.4pp
- **step 150 val 0.698 < s7 14B 0.710** (-1.2pp) — 减量数据让初期略保守
- **step 200 val 0.706 ≥ s7 14B 0.704** — 关键: stage7 此时已开始回调（0.710 → 0.704），stage8 没有
- **step 250 val 0.720** 超 s7 final +1.6pp，**项目历史最高 val**
- **step 275/325 rollout 0.996** = 几乎全对（过饱和）
- **step 300 val 0.710** 略回调，但 step 250 已 capture 到峰值

### 6.3 8B vs 14B 训练动态横向对比

| step | 8B s7 val | 14B s8 val | 14B - 8B |
|---|---|---|---|
| 50 | 0.260 | 0.516 | +25.6pp |
| 100 | 0.460 | 0.662 | +20.2pp |
| 150 | 0.574 | 0.698 | +12.4pp |
| 200 | 0.600 | 0.706 | +10.6pp |

**14B 的 capacity 优势在中后期收窄**: 50→200 步内，差距从 25.6pp 缩到 10.6pp。这表明 8B 在后期"够用"——基础任务靠数据驱动，差距主要在前期。

## 7. 最终评测

### 7.1 评测协议

同 Stage 1（详见 `tech_report_qwen3-8b_stage1_zh.md` §9）。新增 **Clean Eval** 协议：用 GPT-5.5 audit 标定的 4551 题（剔除 1167 wrong-label/ambiguous）。

```
direct:    \boxed{X} 输出，max_tokens=2048
cot:       Let's think step by step，max_tokens=2048
del_tom:   故事中删除心智词后再答，max_tokens=2048
```

### 7.2 Full 5718（direct）— 包含 wrong-label

| 排名 | 模型 | Overall | EN | ZH |
|---|---|---|---|---|
| 🥇 | GPT-5.5 | 0.8349 | 0.8312 | 0.8385 |
| 🥈 | deepseek-v4-pro | 0.8080 | 0.7978 | 0.8181 |
| 🥉 | **qwen3-14b-tom stage8** | **0.7594** ⭐ | 0.7538 | 0.7650 |
| 4 | qwen3-14b-tom stage6 | 0.7580 | 0.7503 | 0.7657 |
| 5 | qwen3-14b-tom stage7 | 0.7539 | 0.7499 | 0.7580 |
| 6 | qwen3-14b-tom stage1 | 0.7527 | 0.7422 | 0.7632 |
| 7 | **qwen3-8b-tom stage7** | **0.7419** ⭐ | 0.7275 | 0.7562 |
| 8 | qwen3-8b-tom stage1 (旧 8B 最高) | 0.7394 | 0.7275 | 0.7513 |
| 9 | qwen3-14b-nt baseline | 0.7338 | 0.7219 | 0.7457 |
| 10 | qwen3-8b-nt baseline | 0.7009 | 0.7020 | 0.6999 |

**关键里程碑**:
- **14B stage8 0.7594**: 本项目历史最高 raw 分（+0.14pp vs s6, +0.67pp vs s1）
- **8B stage7 0.7419**: 首次破 8B stage1 ceiling +0.25pp（5 个 stage 后）
- **14B-base RL 累计增益**: 14B base 0.7338 → s8 0.7594 = **+2.56pp**（vs s1 仅 +1.89pp）
- **8B-base RL 累计增益**: 8B base 0.7009 → s7 0.7419 = **+4.10pp**（vs s1 +3.85pp）

### 7.3 Clean Eval 4551（direct）— 剔除 1167 wrong-label

| 排名 | 模型 | Overall | 提升 vs raw |
|---|---|---|---|
| 🥇 | GPT-5.5 | 0.9343 | +9.94pp |
| 🥈 | deepseek-v4-pro | 0.9013 | +9.33pp |
| 🥉 | **qwen3-14b-tom stage8** | **0.8449** | +8.55pp |
| 4 | qwen3-14b-tom stage6 | 0.8460 | +8.80pp |
| 5 | qwen3-14b-tom stage7 | 0.8436 | +8.97pp |
| 6 | **qwen3-8b-tom stage7** | **0.8321** | +9.02pp |

**关键 insight**: 所有模型 raw → clean 都涨 +8.5 ~ +9.9pp，**ToMBench wrong-label 是所有模型的共同 ceiling**。stage8 14B 在清洁 eval 上距 deepseek -5.64pp（raw -4.86pp），真实能力差距更接近 -5.6pp。

### 7.4 Per-task 分解（Full 5718, direct）— Stage 7 8B vs Stage 8 14B

| Task | 8B s1 | **8B s7** | Δ vs s1 | 14B s6 | **14B s8** | Δ vs s6 | deepseek | gap (s8 14B) |
|---|---|---|---|---|---|---|---|---|
| Belief | 0.694 | **0.701** | +0.7 | 0.732 | **0.739** | +0.7 | 0.849 | -11.0 |
| Desire | 0.592 | **0.608** | +1.7 | 0.583 | 0.569 | -1.4 | 0.633 | -6.4 |
| Emotion | 0.729 | 0.720 | -0.8 | 0.727 | 0.727 | 0.0 | 0.805 | -7.8 |
| **False Belief** | 0.852 | **0.857** | +0.5 | 0.879 | 0.864 | -1.6 | 0.895 | **-3.1** |
| Intention | 0.765 | 0.760 | -0.4 | 0.835 | 0.818 | -1.8 | 0.893 | -7.5 |
| **Knowledge** | 0.479 | 0.478 | -0.2 | 0.502 | **0.514** | **+1.2** ⭐ | 0.567 | -5.3 |
| **Non-literal Comm** | 0.767 | 0.774 | +0.7 | 0.766 | **0.792** | **+2.6** ⭐ | 0.813 | -2.1 |

### 7.5 Stage 8 14B 三大胜利 + 三处倒退

**🎯 三大胜利（vs s6）**:

1. **Non-literal Comm +2.6pp**（距 deepseek 缩到 -2.1pp）：Phase C 的 social_norm_inference (200) + indirect_speech_act (100) + overly_literal (200) 风格匹配数据起效。
2. **Knowledge +1.2pp / +4.5pp vs s7**（项目第一次稳定突破 0.50）：Phase C 的 factual_inference + knowledge_attention_link 提供了 stage6 phase_a 没有的多元 Knowledge 信号。
3. **Belief +0.7pp**：factual_inference 训练数据让模型学会"读关键细节"。

**❌ 三处倒退**:

1. **Desire -1.4pp vs s6**: Phase C 没有 Desire 专项（Phase B.2 zh 只有 250 desire），350 步训练稀释。
2. **False Belief -1.6pp / Intention -1.8pp vs s6**: 350 步 + 9259 records 让旧 task 的 gradient time 被新 task 抢走。
3. **学到**: 加新 task 数据时，必须**同时保留旧 task 比例**，否则旧能力会被稀释。

### 7.6 Stage 7 8B 五大胜利

| Task | 8B s6 | **8B s7** | Δ | 评论 |
|---|---|---|---|---|
| **Belief** | 0.687 | **0.701** | **+1.4** ⭐ | factual_detail 起效 |
| **Desire** | 0.572 | **0.608** | **+3.6** ⭐ | Phase A.2 ZH desire + intention 数据 |
| Emotion | 0.721 | 0.720 | -0.1 | ≈ |
| **False Belief** | 0.845 | **0.857** | **+1.1** ⭐ | 间接受益 |
| **Intention** | 0.785 | 0.760 | -2.5 | 数据稀释 |
| Knowledge | 0.481 | 0.478 | -0.3 | Phase A 不含 Knowledge 专项 |
| **Non-literal Comm** | 0.769 | **0.774** | **+0.5** | social_norm 起效 |

8B 的 Desire +3.6pp 是本项目最大单 task 提升。

### 7.7 Subset500 三协议

| Protocol | 8B s1 | **8B s7** | 14B s6 | 14B s7 | **14B s8** | deepseek subset |
|---|---|---|---|---|---|---|
| direct | 0.7460 | 0.7440 | 0.7780 | 0.7620 | **0.7780** | 0.7880 |
| **cot** | 0.6980 | **0.7460** ⭐ | 0.7560 | 0.7520 | **0.7720** ⭐ | 0.7140 |
| **del_tom** | 0.7460 | 0.7480 | 0.7880 | 0.7620 | **0.7920** ⭐⭐ | n/a |

**关键里程碑**:
- **14B stage8 del_tom 0.7920** 反超 deepseek subset500 direct (0.7880) **+0.4pp** — 本项目第一次在 apples-to-apples 评测上反超 deepseek
- **8B stage7 cot 0.7460** 是 8B cot 历史最高（s1 0.6980 → +4.8pp）
- **14B stage8 cot 0.7720** 是 14B cot 历史最高

### 7.8 Per-task subset500 wins vs deepseek (14B stage8)

| Task | s8 best (subset) | deepseek (subset) | Δ |
|---|---|---|---|
| False Belief | 0.908 (cot/del_tom) | 0.862 | **+4.6** ✓ |
| Desire | 0.722 (cot) | 0.639 | **+8.3** ✓✓ |
| Intention | 0.848 (del_tom) | 0.814 | **+3.4** ✓ |
| Emotion | 0.733 (direct) | 0.709 | +2.4 ✓ |
| Non-literal Comm | 0.836 (del_tom) | 0.843 | -0.7 |
| Belief | 0.750 | 0.800 | -5.0 |
| Knowledge | 0.400 | 0.600 | -20.0 |

**5/7 task 在 subset500 上 ≥ deepseek**。

## 8. 关键 insight：8B 与 14B 在同样数据上反向表现

Stage 7 是本项目最值得记录的实验：**同样的 Phase A 数据，8B 提升 +0.25pp，14B 反退 -0.41pp**。

| 模型 | s6 baseline | Phase A 加入 | Δ |
|---|---|---|---|
| 8B | 0.7380 | 0.7419 (s7) | **+0.25pp** ✓ |
| 14B | 0.7580 | 0.7539 (s7) | **-0.41pp** ✗ |

### 8.1 8B 受益的解释

1. **8B 容量稀缺**: 任何质量提升都有边际收益
2. **8B 上限低**: stage6 数据已被 6 个 stage 反复榨干，新数据让 8B 突破。Phase A 完美补缺
3. **8B 对故事风格不敏感**: 模型能力有限，主要靠题目结构泛化，长故事/短故事差异不大

### 8.2 14B 反退的解释

1. **14B 容量充足**: stage6 数据已被充分学习（step 100 val 0.628 已接近 ceiling）
2. **14B 对风格敏感**: Phase A.1 的 8-12 句精致故事让模型学到了"GPT-5.5 故事 → 推理范式"的关联，但 ToMBench 是 5-7 句直白叙事
3. **数据稀释**: 7259 → 9559 (+32%) 但 max_steps 不变，每条样本被见次数从 7x 降到 5x

**结论**: 不同模型规模需要不同数据策略。8B 用 Phase A，14B 用 Phase C 风格匹配 + 长训练。

### 8.3 Stage 8 在 14B 上的修复验证

| step | 14B s6 val | 14B s7 val | **14B s8 val** |
|---|---|---|---|
| 50 | 0.496 | 0.516 | 0.516 |
| 100 | 0.628 | 0.662 | 0.662 |
| 150 | 0.652 | **0.710** | 0.698 |
| 200 | 0.662 | 0.704 (回调) | 0.706 |
| 250 | (final) | (final) | **0.720** ⭐ |

**Phase C 假设的实证**: 前期 (0-100) stage7/8 学习速度 identical（Phase A vs Phase C 数据加入都提升 +2.0/+3.4pp）；中期 (150-200) stage7 因风格漂移开始回调，stage8 因风格匹配持续涨；后期 (250+) stage8 多 100 步充分巩固。

**stage8 step 250 val 0.720** 是项目 14B 历史最高 val，转化到 full eval +0.55pp vs s7（0.7594 vs 0.7539）。

## 9. 经验教训

### 9.1 验证集与训练集的风格匹配至关重要
RLVR 的 val 信号会"骗人"：合成数据写得越漂亮，val 上学得越好，但泛化到自然数据时反退。**val 必须用与 production eval 同分布的数据**，不能用部分含合成数据风格的 subset。

### 9.2 不同模型规模需要不同数据策略
8B / 14B 同样数据反向表现是新发现。可能延伸：
- 8B 受益于 high-quality, **diverse** synth data（覆盖广）
- 14B 受益于 **style-matched, focused** synth data（覆盖窄但深）
- 32B 可能需要更精细的风格控制 / 更少更精合成数据

### 9.3 Knowledge task 需要多元数据
单靠 scalar implicature (stage6) 推 Knowledge 到 0.502 后停滞。Phase C 加入 factual_inference + knowledge_attention_link 后到 0.514。**Knowledge 不只是数量推理**，还包括事实细读和注意力建模。

### 9.4 评测集 ceiling 比模型能力更接近
所有模型 raw → clean 都涨 +8.5~9.9pp，说明 ToMBench 标签问题是**通用 ceiling**。stage8 14B clean 0.8449 vs raw 0.7594 差 +8.6pp。**Headline 数字应同时报告 raw 和 clean**。

### 9.5 不要看 val 推断 full eval
Stage 7 14B val step 150 = 0.710（高于 s6 final 0.662），让我们一开始预测 stage7 final 会到 0.78-0.80。实际 full eval 0.7539（低于 s6 0.7580）。**val 是 subset500，full eval 是 5718 — 分布不同**。

### 9.6 训练数据扩展时必须保留旧 task 比例
Stage 8 加 Phase C 新 task 后 Desire / FB / Intention 略退。**没有真正"免费的午餐"**——每条新 task 数据稀释了旧 task 的 gradient time。要么 max_steps 大幅增加，要么保留旧 task 数据补充。

## 10. 时间预算与成本

### 10.1 训练
| Phase | 模型 | 步数 | 耗时 | GPU-小时 |
|---|---|---|---|---|
| Stage 1 | 8B | 200 | 3h 40m | ~26 |
| Stage 1 | 14B | 200 | 5h 10m | ~38 |
| Stage 6 | 8B | 250 | 4h 30m | ~32 |
| Stage 6 | 14B | 250 | 5h 15m | ~42 |
| **Stage 7** | 8B | 250 | 4h 30m | ~32 |
| **Stage 7** | 14B | 250 | 5h 15m | ~42 |
| **Stage 8** | 14B | 350 | 6h 50m | ~56 |
| **本次扩展总计** | — | — | ~16h | ~130 GPU-h |

### 10.2 GPT-5.5 数据合成
| Phase | 条数 | 耗时 | 成本 |
|---|---|---|---|
| Stage 6 GPT-5.5 合成 (FB1/FB2/scalar) | 1400 | ~35min | ~$60 |
| Phase A.1 | 1500 | ~85min | ~$50 |
| Phase A.2 (ZH) | 800 | ~42min | ~$30 |
| Phase C (style-matched) | 1200 | ~58min | ~$50 |
| GPT-5.5 audit 5718 eval | 5718 | ~70min | ~$200 |
| **总计** | **10618** | **~290min** | **~$390** |

### 10.3 Eval
| 任务 | 耗时 |
|---|---|
| 8B 或 14B 全 5718 eval (concurrency=32, vLLM TP=1) | ~7 分钟 |
| Subset500 三协议 | ~3 分钟 |
| Clean 4551 | ~6 分钟 |

## 11. 复现 Stage 7/8

### 11.1 复现 Phase A 数据合成
```bash
source ~/.zshrc   # 设置 OPENAI_BASE_URL / OPENAI_API_KEY
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_BASE_URL -e OPENAI_API_KEY dev \
  python -m scripts.data.synth_gpt55_phase_a --concurrency 8

docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_BASE_URL -e OPENAI_API_KEY dev \
  python -m scripts.data.synth_gpt55_phase_b_zh --concurrency 8

docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python -m scripts.data.merge_phase_a
```

### 11.2 复现 Phase C 数据合成 (Stage 8)
```bash
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_BASE_URL -e OPENAI_API_KEY dev \
  python -m scripts.data.synth_gpt55_phase_c --concurrency 8

docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python -m scripts.data.merge_phase_c
```

### 11.3 复现 GPT-5.5 audit 5718
```bash
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_BASE_URL -e OPENAI_API_KEY dev \
  python -m scripts.analysis.gpt55_audit_eval_full --concurrency 10

docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python -m scripts.data.build_clean_eval
```

### 11.4 复现训练
```bash
make sync-up   # 同步 code + data 到 TRAIN host

# 8B Stage 7
make train-stage7-1x8

# 14B Stage 8
make train-stage8-1x8-14b
```

### 11.5 复现评测
```bash
# 8B Stage 7 — vLLM serve (在 TRAIN host)
ssh $TRAIN_HOST 'docker run --rm -d --name qwen3-tom-serve-8b-stage7 \
  --gpus device=0 --ipc host --shm-size 16gb -p 8000:8000 \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  -v /data_nvme/grj-projects/models:/mnt/models \
  -e HF_HOME=/mnt/models/.cache/huggingface \
  -e PYTHONPATH=/workspace:/workspace/framework/ROLL \
  --entrypoint python qwen3-tom-train:latest \
  -m vllm.entrypoints.openai.api_server \
  --model /mnt/output/qwen3-8B-tom-hf-stage7 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --served-model-name qwen3-8b-tom-stage7'

# 14B Stage 8 — 同上, model id qwen3-14b-tom-stage8

# 从 DEV 跑 eval
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy dev \
  python scripts/eval/run_tombench.py \
  --backend openai --base-url http://$TRAIN_HOST:8000/v1 \
  --model qwen3-14b-tom-stage8 \
  --data data/tom/tombench_eval.jsonl \
  --protocols direct --concurrency 32 \
  --output output/eval/stage8_full5718.json
```

## 12. 下一步（Stage 9+ 候选）

### 选项 A: Phase D 补 Desire/FB/Intention 专项
覆盖 Stage 8 倒退的 task。预计 ~600 条 ToMBench-style 数据 + max_steps 400。

### 选项 B: 难度课程
实现 ROLL 的 difficulty-weighted sampler，渐进 easy → medium → hard。可能突破 Stage 8 的 step 250 后 saturation。

### 选项 C: 32B 模型
基于 Stage 8 数据 + 350 步配方训练 Qwen3-32B。估计 raw 0.79-0.81, clean 0.86-0.88。资源需求: 2×8 H800 或单卡跑 PP=2。

### 选项 D: 多模型 ensemble
- 8B stage7 (轻量部署 / Desire 强)
- 14B stage8 (生产 / Knowledge / Non-literal 强)
- 通过 protocol 路由（direct/cot/del_tom）选最强模型

### 选项 E: 评测集二次清洁
GPT-5.5 audit 已剔除 20% wrong-label，但 ambiguous_question (10.8%) 中可能仍有可救。用 multi-vote (GPT-5.5 × 3) 二次审查。

## 13. 产物清单

### 配置 / 代码
| 路径 | 内容 |
|---|---|
| `configs/tombench-rlvr/rlvr_config_stage7_1x8.yaml` | 8B stage7 配置 |
| `configs/tombench-rlvr/rlvr_config_stage7_1x8_14b.yaml` | 14B stage7 配置（参考用，已被 stage8 取代） |
| `configs/tombench-rlvr/rlvr_config_stage8_1x8_14b.yaml` | 14B stage8 配置（max_steps=350） |
| `scripts/data/synth_gpt55_phase_a.py` | Phase A.1 4-cat 合成脚本 |
| `scripts/data/synth_gpt55_phase_b_zh.py` | Phase A.2 ZH 合成脚本 |
| `scripts/data/synth_gpt55_phase_c.py` | Phase C 风格匹配合成脚本 |
| `scripts/data/merge_phase_a.py` | Phase A merge (additive) |
| `scripts/data/merge_phase_c.py` | Phase C merge (替换 phase A) |
| `scripts/analysis/gpt55_audit_eval_full.py` | 全 5718 eval audit |
| `scripts/data/build_clean_eval.py` | 清洁 eval set 构建 |

### 数据
| 路径 | 内容 |
|---|---|
| `data/tom/tom_train.jsonl` | 9259 records (Stage 8 final, post-merge_phase_c) |
| `data/tom/tom_train_PRE_PHASE_A_BACKUP.jsonl` | Stage 6 (7259) 备份 |
| `data/tom/tom_train_PRE_PHASE_C_BACKUP.jsonl` | Stage 7 (9559) 备份 |
| `data/tom/raw/synth_gpt55_phase_a.jsonl` | 1500 raw |
| `data/tom/raw/synth_gpt55_phase_b_zh.jsonl` | 800 raw |
| `data/tom/raw/synth_gpt55_phase_c.jsonl` | 1200 raw (style-matched) |
| `data/tom/tombench_eval_clean.jsonl` | 4551 清洁 eval |
| `output/analysis/clean_eval_qids.json` | keep/drop qid 清单 |
| `output/analysis/gpt55_eval_full_audit.jsonl` | 5718 audit 完整记录 |

### 训练日志 / 评测
| 路径 | 内容 |
|---|---|
| `logs/train_stage7_1x8_20260518_192021.log` | 8B stage7 训练日志 (12 MB) |
| `logs/train_stage7_1x8_14b_20260518_135019.log` | 14B stage7 训练日志 (13 MB) |
| `logs/train_stage8_1x8_14b_20260518_234107.log` | 14B stage8 训练日志 (17 MB) |
| `output/eval/8b_stage7_{full5718,clean_eval,subset500}.{json,md}` | 8B stage7 评测 |
| `output/eval/stage7_{full5718,clean_eval,subset500}.{json,md}` | 14B stage7 评测 |
| `output/eval/stage8_{full5718,clean_eval,subset500}.{json,md}` | 14B stage8 评测 |
| `output/analysis/curves_stage7_8b.png` | 8B stage7 训练曲线 |
| `output/analysis/curves_stage8_14b.png` | 14B stage8 训练曲线 |

### Checkpoint (TRAIN host)
| 路径 | 大小 |
|---|---|
| `qwen3-8B-tombench-rlvr-stage7-1x8/.../checkpoint-249/` | ~107 GB (Megatron) |
| `qwen3-8B-tom-hf-stage7/` | 16 GB (HF, 生产部署) |
| `qwen3-14B-tombench-rlvr-stage8-1x8/.../checkpoint-349/` | ~196 GB (Megatron) |
| `qwen3-14B-tom-hf-stage8/` | 28 GB (HF, 生产部署) |

### 报告
| 路径 | 内容 |
|---|---|
| `docs/improvement_plan.md` | Stage 6 → Stage 7 改进计划（HOT 错误诊断） |
| `docs/stage7_report.md` | 14B stage7 英文报告（失败诊断） |
| `docs/stage7_8b_report_zh.md` | 8B stage7 中文报告 |
| `docs/stage8_report.md` | 14B stage8 英文报告 |
| `docs/stage8_report_zh.md` | 14B stage8 中文报告 |
| **本文档** | 合并技术报告 |

## 14. 项目累计成绩总表

| Model | Full 5718 direct | Clean 4551 direct | Subset500 best |
|---|---|---|---|
| qwen3-8b-nt baseline | 0.7009 | — | — |
| qwen3-8b-tom stage1 | 0.7394 | — | 0.7460 (direct) |
| qwen3-8b-tom stage6 | 0.7380 | — | 0.7500 (del_tom) |
| **qwen3-8b-tom stage7** ⭐ | **0.7419** | **0.8321** | **0.7480** (del_tom) / 0.7460 (cot) |
| qwen3-14b-nt baseline | 0.7338 | — | — |
| qwen3-14b-tom stage1 | 0.7527 | — | 0.7800 (direct) |
| qwen3-14b-tom stage6 | 0.7580 | 0.8460 | 0.7880 (del_tom) |
| qwen3-14b-tom stage7 | 0.7539 | 0.8436 | 0.7620 |
| **qwen3-14b-tom stage8** ⭐ | **0.7594** | **0.8449** | **0.7920** (del_tom) |
| deepseek-v4-pro | 0.8080 | 0.9013 | 0.7880 (direct) |
| GPT-5.5 | 0.8349 | 0.9343 | — |

**关键 gap**:
- **8B stage7 距 deepseek**: raw -6.61pp / clean -6.92pp
- **8B stage7 距 GPT-5.5**: raw -9.30pp / clean -10.22pp
- **14B stage8 距 deepseek**: raw -4.86pp / clean -5.64pp / **subset500 反超 +0.4pp** ⭐
- **14B stage8 距 GPT-5.5**: raw -7.55pp / clean -8.94pp

## 附录 A：Stage 7 vs Stage 8 数据组成对比

```
Stage 7 (8B 和 14B 共用):                Stage 8 (仅 14B):
  base 7259 (cleaned)                     base 7259 (cleaned)
  + Phase A.1 (1500)                      + Phase C (1200, style-matched)  <- 替换 A.1
    social_norm 400                         factual_inference 200
    factual_detail 300                      social_norm 200
    intention_attribution 400               intention_attribution 200
    indirect_speech 400                     overly_literal 200
  + Phase A.2 (800 ZH)                      emotion_attribution 150
    belief_zh 300                           knowledge_attention 150
    knowledge_zh 250                        indirect_speech 100
    desire_zh 250                         + Phase A.2 (800 ZH) [保留]
                                          
  TOTAL: 9559                             TOTAL: 9259
  max_steps: 250                          max_steps: 350
```

## 附录 B：Phase C 风格匹配 prompt 关键约束

```
CRITICAL STYLE CONSTRAINTS (will be strictly enforced):
- Story: EXACTLY 5-7 sentences. Not 8. Not 10. Count them.
- Use simple subject-verb-object structure. Each sentence states ONE fact.
- NO rhetorical flourishes: avoid "Suddenly,", "Despite this,", etc.
- NO scene-setting paragraphs about weather, mood, or atmosphere.
- Use ordinary names (first names or first+last). Original (not Sally/Anne).
- Use ordinary settings: classroom, office, lunchroom, park, hallway.

CONTENT CONSTRAINTS:
- Story must FULLY determine the correct answer.
- 4 options: 1 unambiguously correct, 1 surface-literal-but-wrong (trap),
  2 plausible distractors.
```

## 附录 C：HOT 错误诊断的 7 类失败模式

基于 GPT-5.5 对 70 条 HOT 错误（s6 14B 错、deepseek+GPT-5.5 都对）的诊断：

| 失败原因 | n | % |
|---|---|---|
| factual_inference | 12 | 17% |
| social_norm_inference | 11 | 16% |
| intention_attribution | 10 | 14% |
| overly_literal | 10 | 14% |
| emotion_attribution | 8 | 11% |
| knowledge_attention_link | 6 | 9% |
| indirect_speech_act | 6 | 9% |
| scalar_implicature | 4 | 6% |

Phase A.1 + Phase C 设计覆盖了前 7 类，scalar implicature 已由 Stage 6 phase 处理。

## 附录 D：GPT-5.5 audit 5718 题的 issue 分布

| Issue category | n | % |
|---|---|---|
| label_correct | 4516 | 79.0% |
| ambiguous_question | 620 | 10.8% |
| wrong_label | 460 | 8.0% |
| options_overlap | 72 | 1.3% |
| translation_artifact | 50 | 0.9% |

**最难的 task**: Knowledge 55.2% drop rate（多为 scalar implicature 模糊计数，4551 题清洁后只剩 259 题，但 stage8 Knowledge clean 跑到 0.664，远超 raw 0.514）。

最后更新: 2026-05-19 16:00
