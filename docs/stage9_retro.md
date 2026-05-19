# Stage 9 Retrospective: SFT Cold Start Did Not Work on ToMBench

> 训练: 2026-05-19 (SFT 14B 17 min + GRPO 14B 16h 15min)
> 评测: 2026-05-20 13:00 UTC (full 5718 + clean 4551 + subset500 × 3 protocols)
> Result: **All protocols regressed vs stage 8**

## 1. 核心结果

| Eval | Stage 8 (best) | **Stage 9** | Δ |
|---|---|---|---|
| **Full 5718 direct** | **0.7580** | **0.7429** | **-1.51pp** ↓ |
| Clean 4551 direct | 0.8449 | 0.8361 | -0.88pp |
| Subset500 direct | 0.7780 | 0.7380 | -4.0pp |
| Subset500 cot | 0.7720 | 0.7500 | -2.2pp |
| Subset500 del_tom | **0.7920** | 0.7600 | -3.2pp |

**Stage 9 完全没有击败 Stage 8。Plan v3 Phase 1 假设需要重新审视**。

## 2. 训练流程

### 2.1 SFT cold start (Stage 9a)
- 14B base + 3730 GPT-5.5 reasoning traces
- 174 步, 17 分钟训练
- Loss: 0.55 → 0.27 (健康下降)
- 3 ckpts saved (50/100/150 = HF format)

### 2.2 GRPO (Stage 9b)  
- Init from SFT ckpt-150
- 350 步, 16h 15min
- val 趋势:
  - step 50: 0.718 (+20pp vs s8 同期 0.516) ⭐
  - step 100: 0.712
  - step 150: 0.706
  - step 200: 0.736 ⭐ (peak)
  - step 250: 0.732
  - step 300: 0.734
  - step 349 (final): 估计 0.73-0.74

**val 远超 stage8 同期 (+3-20pp)，但 full eval 反而退步 -1.5pp**。

## 3. 失败根因分析

### 3.1 标签冲突问题

我们 SFT prompt 教模型用 `<reasoning>...</reasoning>` 标签。但 **Qwen3 base 模板自带 `<think>...</think>` 标签机制**。SFT 后样本输出：

```
<think>

</think>

<推理段落>

$$
\boxed{X}
$$
```

**模型空了 `<think>` 标签，把推理放在标签外**。这是 chat_template 与 SFT 教法的冲突。

### 3.2 SFT 模板教模型在所有题上写长推理，但简单题不需要

ToMBench 题型分析：
- ~60% 题在 stage8 上 0.7580 已经答对（直接 1-2 步推断）
- ~40% 题需要多步推理

SFT 让模型在**所有 100% 题上**都写 4-6 步推理。**简单题被想复杂了**，反而引入推理错误：
- 错样本 "Xiao Mei sees Xiao Hong and Xiao Fang exchange a meaningful glance" → 推理 4 步走偏到选 A（正确 B）
- stage8 直接走 1 步推断，反而对

### 3.3 lr 5e-7 太保守 / 实际上更激进

- 我们用 lr 5e-7 (vs s8's 1e-6) 来"避免 SFT 遗忘"
- 但实际上: SFT 是 fresh fine-tune base, 没什么需要 protect 的 RL 进度
- lr 5e-7 + 350 步 RL 信号弱, 很难真正改进 SFT 学到的推理风格

### 3.4 GRPO val (subset500 with reasoning encouraged) 与 full eval (direct, prefer succinct) 的协议不匹配

stage9 val_correct/all 在 subset500 上 0.736 — 但用 cot 协议
stage9 full 5718 用 direct 协议 (max_tokens=2048, 但模型用了 reasoning chain)
**val 上 reasoning 模板获益, full eval 上未必**

### 3.5 Per-task 退步分析

vs s8:

| Task | s8 | s9 | Δ |
|---|---|---|---|
| Belief | 0.732 | 0.694 | **-3.8** ↓ |
| Desire | 0.583 | 0.597 | +1.4 ✓ |
| Emotion | 0.727 | 0.716 | -1.1 |
| False Belief | 0.879 | 0.847 | **-3.2** ↓ |
| **Intention** | **0.835** | **0.754** | **-8.1** ↓↓↓ |
| Knowledge | 0.514 | 0.505 | -0.9 |
| Non-literal | 0.792 | 0.786 | -0.6 |

**Intention -8.1pp** 是最大退步。Intention 题型需要**简短推断 → 选最佳意图**，强制 4-6 步推理把简单的"识别意图"问题想成了多步推断链。

## 4. 实际有效的训练方法（重新评估）

| 方法 | s9 实测结果 | 总结 |
|---|---|---|
| SFT cold start | **不 work**（-1.5pp） | ToMBench 题型短，SFT 长 CoT 反而引入错误 |
| Dr.GRPO loss | 中性 | 训练曲线似乎平稳 |
| Long CoT (resp 1024) | **可能负面** | Direct 协议 val_max_tokens=512 仍然限制 |
| KL=0.001 + entropy=0.01 | 中性 | 训练稳定但没"激活" SFT 进步 |
| Weighted-sum reward | **可能正面** | 但被其他负面因素掩盖 |

## 5. 改回 Stage 8 配置 + 局部改进

最务实的下一步：
1. **Stage 10**: 回到 stage8 配方 (no SFT, 9259 records, 350 steps, response 256)
2. **加 weighted-sum reward** (s8 用 multiplicative)
3. **加 entropy bonus 0.01**
4. **微小 KL 0.001** (主要为稳定)
5. **不动 difficulty mask**, 不加 long CoT
6. **不做 SFT**

预期: stage10 raw ~0.76-0.77（约 +0.5-1pp vs s8）。

## 6. SFT 在哪些场景是 work 的

参考 DeepSeek-R1 / Light-R1 文献：
- 数学题（AIME / MATH）: 题需要长 CoT, SFT 大幅 +5-10pp
- 编程题: 类似
- **ToMBench**: **不 work**, 因为题型短 + SFT 教的多步推理引入 noise

ToM 任务是 reasoning depth shallow + breadth wide 的形态，**不需要**纯 CoT-style SFT。

## 7. 后续 Plan v3 战略调整

**取消** SFT cold start 路线。改为:
- **Stage 10**: stage8 配方 + weighted-sum reward + entropy bonus + 极低 KL（不带 SFT）
- **Stage 11**: ExploreToM 程序化数据（v3 Phase 3）
- **Stage 12**: 难度课程（v3 Phase 2，不依赖 SFT）

预期最终 14B 上限 0.78-0.81 (without SFT)。

## 8. 产物清单

| 路径 | 内容 |
|---|---|
| `output/eval/stage9_full5718.{json,md}` | full 5718 (0.7429) |
| `output/eval/stage9_clean_eval.{json,md}` | clean 4551 (0.8361) |
| `output/eval/stage9_subset500.{json,md}` | subset500 × 3 |
| `data/tom/raw/reasoning_traces.jsonl` | 3830 GPT-5.5 traces |
| `data/tom/tom_train_sft.jsonl` | 3730 SFT records |
| `data/tom/tom_train_sft_val.jsonl` | 20 val |
| `output/analysis/trace_audit_sample100.jsonl` | 100-sample audit |
| `logs/train_sft_stage9_14b_20260519_124127.log` | SFT log (14 MB) |
| `logs/train_stage9_1x8_14b_20260519_132823.log` | GRPO log (18 MB) |
| HF model | `qwen3-14B-tom-hf-stage9/` |
| Megatron ckpt | `qwen3-14B-tombench-rlvr-stage9-1x8/.../checkpoint-349/` |
| SFT HF | `qwen3-14B-tom-sft-stage9/<timestamp>/checkpoint-150/` |

最后更新: 2026-05-20 14:00
