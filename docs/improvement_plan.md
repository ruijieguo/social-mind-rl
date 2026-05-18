# 训练与评测深度分析与改进建议

> 基于 stage1-stage6 全部 8B/14B 训练 + GPT-5.5 数据审查 + 三方错误分解 + HOT 错误 GPT-5.5 诊断的综合分析。
>
> 目标: 8B 突破 0.74 ceiling, 14B 突破 0.78, 双模型逼近 GPT-5.5 (0.8349)。

## 1. 当前所有模型分数全景

**Full ToMBench 5718 direct:**

| Model | Overall | Belief | Desire | Emotion | False Belief | Intention | Knowledge | Non-literal |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b-nt | 0.7009 | 0.673 | 0.586 | 0.689 | 0.728 | 0.750 | 0.481 | 0.777 |
| qwen3-8b-tom stage1 | 0.7394 | 0.694 | 0.592 | 0.729 | 0.852 | 0.765 | 0.479 | 0.767 |
| qwen3-8b-tom stage6 | 0.7380 | 0.687 | 0.572 | 0.721 | 0.845 | 0.785 | 0.481 | 0.769 |
| qwen3-14b-nt | 0.7338 | 0.736 | 0.583 | 0.720 | 0.805 | 0.766 | 0.467 | 0.795 |
| qwen3-14b-tom stage1 | 0.7527 | 0.747 | 0.589 | 0.729 | 0.877 | 0.810 | 0.478 | 0.764 |
| **qwen3-14b-tom stage6** | **0.7580** | 0.732 | 0.583 | 0.727 | 0.879 | 0.835 | **0.502** | 0.766 |
| **deepseek-v4-pro** | **0.8080** | 0.849 | 0.633 | 0.805 | 0.895 | 0.893 | 0.567 | 0.813 |
| **GPT-5.5** | **0.8349** | 0.842 | 0.681 | 0.815 | 0.926 | 0.879 | 0.671 | 0.834 |

**Key gap to close**: 14B-stage6 (0.7580) → GPT-5.5 (0.8349) = **−7.69pp**
**8B 上限**: stage1 0.7394（5 个 stage 配方均收敛此处）

## 2. 三方错误分解 (核心新洞察)

把 14B-stage6 的 1384 条错误按"GPT-5.5 + deepseek 是否同时正确"分类:

| 错误类型 | n | 占比 | 含义 |
|---|---|---|---|
| **all_three_wrong** (硬上限) | 557 | 40% | 三家都错 → 标签噪声 / 真模糊 / 不可优化 |
| only_gpt5_right | 225 | 16% | 只有 GPT-5.5 对 → 难，但有 ceiling |
| only_ds_right | 133 | 10% | 只有 deepseek 对 → 同上 |
| **both_right_14b_wrong (HOT)** | **469** | **34%** | 两家都对，14B 错 → **可优化** |

**HOT 469 条 = +8.2pp 上限**（如全部修复，14B-stage6 就能从 0.758 → 0.840，已超 deepseek，逼近 GPT-5.5）。

每 task 的 HOT 比例:

| Task | 总错误 | HOT (高价值修复) | hard ceiling |
|---|---|---|---|
| Belief | 76 | **47%** | 27% |
| False Belief | 179 | **44%** | 25% |
| Intention | 112 | **42%** | 27% |
| Non-literal Comm | 350 | 34% | 37% |
| Emotion | 229 | 33% | 41% |
| Knowledge | 288 | 25% | **53%** ⚠ |
| Desire | 150 | 22% | **52%** ⚠ |

**Knowledge 和 Desire 的硬上限 > 50%**: 标签问题严重，单纯多训没用。**Belief / FB / Intention 是高 ROI 修复区**: 错误中 40-47% 都是"应该会但没会"的可学型。

## 3. HOT 错误的失败原因诊断 (GPT-5.5 解释 70 条样本)

| 失败原因 | n | % | 根因 |
|---|---|---|---|
| **factual_inference** | 12 | 17% | 故事关键细节没读到（细读能力） |
| **social_norm_inference** | 11 | 16% | 社交常识：失礼/送礼/工作场景规范 |
| **intention_attribution** | 10 | 14% | 把行为归因到错的目标 |
| **overly_literal** | 10 | 14% | 字面解读，错过 nonverbal/暗示 |
| **emotion_attribution** | 8 | 11% | 复杂或冲突情绪推断（如赢得比赛 → 高兴） |
| **knowledge_attention_link** | 6 | 9% | 谁看到了什么 + 推到 belief |
| **indirect_speech_act** | 6 | 9% | 间接请求（"嘴干"想要奶茶） |
| **scalar_implicature** | 4 | 6% | "几乎没有 ≈ 1-3" |
| 其他 | 3 | 4% | desire / temporal |

**关键 insight**: 我们 5 stage 训练**严重偏 false belief 范式**（exploretom + simpletom 都是 sally-anne 模板），但实际 HOT 错误中只有 ~9% 是 "first/second-order belief"。剩下 91% 是其他类型——其中 **31% 是"读细节 + 社交常识"**，这两个我们的训练数据基本没覆盖。

## 4. 训练动态对比 (14B 和 8B 在同样数据下的关键差异)

| step | 14B-s6 score | 8B-s6 score | 14B all_correct | 8B all_correct |
|---|---|---|---|---|
| 50 | **0.717** | 0.335 | 0.25 | 0.06 |
| 100 | **0.957** | 0.758 | **0.84** | 0.25 |
| 150 | **0.936** | 0.835 | 0.91 | 0.44 |
| 200 | **0.973** | 0.851 | 0.91 | 0.69 |
| 225 | 0.932 | 0.878 | 0.84 | 0.62 |

**14B step 100 已 saturate** (`all_correct=0.84`, samples_used=16/256, 99.2% 数据被难度遮罩丢弃)
**8B step 200 还在学** (`all_correct=0.69`, samples_used=48/256, 81% 被遮罩)

**信号完全不同的两件事**：
- 14B 后期 200 步是**对训练集死记硬背**，泛化天花板已经触达，多训就是纯过拟合
- 8B 后期还有 ~20% 信号，但内容本身是"中等难度"题目；高难度（社交常识、间接言语）已被难度遮罩丢光，模型根本学不到这部分

## 5. EN vs ZH 关键观察

| Task | 14B-s6 EN | 14B-s6 ZH | gpt EN | gpt ZH | s6 vs gpt EN | s6 vs gpt ZH |
|---|---|---|---|---|---|---|
| Belief | 0.711 | 0.754 | 0.803 | 0.880 | -9.2pp | **-12.7pp** |
| Emotion | 0.683 | 0.771 | 0.798 | 0.833 | **-11.4pp** | -6.2pp |
| Knowledge | 0.522 | 0.481 | 0.678 | 0.664 | -15.6pp | **-18.3pp** |
| Desire | 0.594 | 0.572 | 0.650 | 0.711 | -5.6pp | **-13.9pp** |

**ZH 上 gap 平均比 EN 大 3-5pp**——但 14B 自己 ZH 得分大多 > EN，说明不是模型的中文能力差，是中文训练数据特别薄。我们的训练数据 EN/ZH ≈ 65/35。

## 6. 改进策略矩阵 (按预期 ROI 排序)

### A. 数据扩展（决定性）

#### A.1 [极高 ROI] 合成 social_norm + factual + intention 类数据 1500 条

当前训练数据完全没覆盖 GPT-5.5 诊断的 top-3 失败类型 (50% HOT 错误)。

具体合成（用 GPT-5.5）：
- **400 条 social_norm**: 失礼识别（送礼/餐桌/职场）+ 礼貌性反应识别。重点训练 "对方礼貌微笑 ≠ 没失礼" 的反向归因
- **300 条 factual_detail**: 故事中关键非显眼细节驱动答案（如"她离开后又回来"导致 belief 状态变化），训练细读
- **400 条 intention_attribution**: 角色行为 + 隐藏目标推理（销售员的夸张话术驱动"想推销"，不是字面笑话）
- **400 条 indirect_speech_act**: 暗示请求识别（"嘴干"=想喝水/奶茶；"今天好热啊"=请开空调）

**预期增益**: 修复 30-40% HOT 错误 → 14B 总分 +2.5-3.5pp，到 0.78-0.79（接近 deepseek）

#### A.2 [高 ROI] 中文数据扩充

当前 ZH 训练样本 ~3500，少于 EN ~3700 的 5%，但 ZH gap 比 EN 大 ~5pp。

合成 800 条 GPT-5.5 中文 ToM 数据，重点 Belief / Knowledge / Desire (ZH gap 最大的 3 个 task)。

**预期增益**: ZH 总分 +1.5pp（占 总分 0.5%）→ Overall +0.5-1.0pp

#### A.3 [中 ROI] Knowledge / Desire 不再追加

GPT-5.5 audit 显示这两个 task hard_ceiling > 50%，再投数据边际收益低。Knowledge 已用 GPT-5.5 scalar 数据涨了 +2.42pp on 14B, 边际效益已显现。

### B. 训练算法（突破 8B 上限）

#### B.1 [核心改进] 难度自适应数据课程 (curriculum)

当前问题: 14B step 75 后 90%+ 训练样本被难度遮罩丢掉，没在学**真正难的题**。

方案：
1. 训练前评测 base model 在每条训练样本上的准确率（10 样本投票）
2. 把样本分 3 桶: easy (>80%), medium (40-80%), hard (<40%)
3. 训练阶段:
   - step 0-50: 50% easy + 40% medium + 10% hard (建立格式)
   - step 50-150: 20% easy + 50% medium + 30% hard (深入推理)
   - step 150+: 10% easy + 30% medium + 60% hard (突破上限)

8B 这样可能突破 0.74 ceiling, 14B 不会过早 saturate。

**预期增益**: 8B +0.5-1.5pp, 14B +1-2pp

#### B.2 [实验性] 双模型 group-relative 奖励

GRPO 现在用同模型 8 样本对比，但 saturate 后所有样本都对，advantage 趋零。改成: **每组用同 prompt 让 14B-s6 + 14B-base + GPT-5.5 各采样**，让训练样本始终包含挑战性案例。

风险: 改动较大，需要修改 ROLL 框架。

**预期增益**: 14B +1-2pp，但实施复杂

#### B.3 [低 ROI] response_length 调整

stage3 验证过 384 反而坏（val 协议截断到 64 tokens）。保持 256 即可。

### C. 评测协议改进

#### C.1 [必做] 清理 wrong-label 评测题

GPT-5.5 audit 在 196 个 both-wrong 样本中发现 40% 是 wrong_label。**展开到全 5718**：
1. 用 GPT-5.5 audit 全部 5718 题 (~$300, 2 小时)
2. 标 wrong_label 高置信度的题，从 eval set 中排除
3. 在"清洁 eval"上重测所有模型

**预期影响**: 所有模型分数 +2-4pp。14B-s6 修正后估计 ≈ 0.79-0.81, deepseek ≈ 0.84, GPT-5.5 ≈ 0.87。**真实 gap 比账面更小**。

#### C.2 [必做] subset500 → 全量 5718 一致性

del_tom 协议只在 subset500 上跑过，stage6 14B 在 subset500 是 0.788，**应在全量 5718 重测**确认。

### D. 模型规模

#### D.1 [终极方案] Qwen3-32B 训练

如果资源允许: 32B + cleaned data + GPT-5.5 synth 配方。基于 8B → 14B 的 +1.89pp RL 收益和 +3.29pp size 收益规律外推:
- 32B-base 估计 0.76+
- 32B-tom stage6 估计 0.78-0.80, 接近 deepseek-pro

资源需求: 1×8 H800 TP=4 + DP=2 (14B 是 TP=2 + DP=4)，每步 ~2× 慢。

#### D.2 [优先] 双 8 卡节点跑 14B

14B 现在 TP=2 + DP=4。换成 2×8 → TP=2 + DP=8 (rollout 大 2 倍)，单步等长但 rollout 更多样化，更可能避免难度遮罩饱和。

## 7. 建议执行优先级

### Phase A (1 周): 数据 + 评测整治
1. ✅ **A.1** GPT-5.5 合成 1500 条专项数据 (social_norm/factual/intention/indirect_speech_act)
2. ✅ **C.1** GPT-5.5 audit 全 5718 评测，输出清洁 eval set
3. ✅ **A.2** GPT-5.5 合成 800 条中文 ToM 数据

成本: GPT-5.5 API ~$500，时间 ~6h

### Phase B (1 周): 训练
4. **B.1** 实现难度课程: 在 ROLL 数据加载里加一个 difficulty-weighted sampler
5. 重训 14B stage7: 7259 现有 cleaned + 2300 新 GPT-5.5 数据 + 难度课程, 250 步
6. 重训 8B stage7: 同样数据，看 8B 是否能突破 0.74

预期 stage7 结果:
- 14B-stage7: 0.78-0.80 (账面), 清洁 eval 上 0.83+
- 8B-stage7: 0.74-0.76 (账面)

### Phase C (2 周): 终极尝试
7. **D.2** 2×8 H800 跑 14B-stage8 (大 rollout, 更多步数, 难度课程)
8. **D.1** 如有资源跑 Qwen3-32B-stage6

预期最终:
- 14B-stage8: 0.80+ (账面), 清洁 eval 0.84+ ≈ deepseek
- 32B-stage6: 0.79-0.81 (账面)

## 8. 可立即开始的最小动作 (Phase A.1)

新合成数据规格 (GPT-5.5):

```python
SYNTH_CATEGORIES = {
    "social_norm": {
        "n": 400,
        "task_tag": "Non-literal Comm",
        "prompt_focus": "Faux-pas + polite reaction inversion",
        "key_constraint": "model trained on this should NOT confuse 'listener didn't object' with 'no faux-pas occurred'",
    },
    "factual_detail": {
        "n": 300,
        "task_tag": "Belief|False Belief",
        "prompt_focus": "Story has 1-2 critical detail sentences (e.g., 'X left the room', 'Y returned at 3pm') that are NOT in the question; correct answer requires noticing these",
    },
    "intention_attribution": {
        "n": 400,
        "task_tag": "Intention|Desire",
        "prompt_focus": "Character action + hidden goal + observer's most-likely-inference. NOT direct speech.",
    },
    "indirect_speech_act": {
        "n": 400,
        "task_tag": "Intention",
        "prompt_focus": "Hint patterns: '今天嘴有点干' (want drink), '路边好多花' (compliment to companion), '楼下有家咖啡馆' (social invite). Make options have 1 indirect-meaning + 1 literal + 2 distractors.",
    },
}
```

这 1500 条单独跑 GPT-5.5 (concurrency=8) 约需 1 小时, $400 左右。

合成完后:
- 跑 leakage check vs ToMBench eval (≥0.6 Jaccard 全 drop)
- merge 进 tom_train.jsonl → 总 ~8800
- 训练 14B stage7 (250 步) + 8B stage7 (250 步)
- 全量 5718 + subset500 评测

## 9. 总结结论

1. **8B 上限是数据 + capacity 的双重 ceiling**: 5 stage 不同配方收敛在 0.74，没法靠重新组合既有元素突破。需要质量更高的数据 + 难度课程
2. **14B 还有显著空间**: HOT 错误 469 条 = +8.2pp 上限。但当前训练 step 100 已 saturate，模型在反复学已经会的，不学不会的
3. **真正缺的是 social_norm + factual_detail + intention 类数据**: 这是 GPT-5.5 比 deepseek 强的方向（Knowledge +10.4pp, Desire +4.7pp, FB +3.2pp 都对应这些类别），也是我们 90% 错误的根源
4. **评测集本身是另一个 ceiling**: 14B-stage6 错误的 40% 是三家都错的硬上限，标签问题严重；清理后所有模型 +2-4pp
5. **下一步明确**: GPT-5.5 合成 1500 + 800(zh) 条专项数据 + 清理 eval set + 难度课程，预计 14B 到 0.78-0.80，8B 到 0.74-0.76

撰写完成: 2026-05-18 12:50
