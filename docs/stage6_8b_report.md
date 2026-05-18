# Stage 6 报告：Qwen3-8B + 清洁数据 + GPT-5.5 合成

> 训练: 2026-05-18 07:36 → 11:57 (UTC)；250 步, 7259 清洁后训练数据
> 评测: 2026-05-18 12:18（全量 5718 + subset500 × 3 协议）
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage6-1x8/20260517-233202/checkpoint-249/`

## 1. 关键结果

**Full ToMBench 5718 (direct)**:

| 模型 | Overall | Δ vs 8B baseline |
|---|---|---|
| qwen3-8b-nt | 0.7009 | — |
| **qwen3-8b-tom stage1 (旧 8901 数据)** | **0.7394** | +3.85pp ← 8B 历史最高 |
| qwen3-8b-tom stage2 | 0.7263 | |
| qwen3-8b-tom stage3 | 0.7302 | |
| qwen3-8b-tom stage5 | 0.7305 | |
| **qwen3-8b-tom stage6 (清洁 + GPT-5.5)** | **0.7380** | +3.71pp |
| qwen3-14b-tom stage6 | 0.7580 | +5.71pp vs 8B |
| deepseek-v4-pro | 0.8080 | — |
| GPT-5.5 | 0.8349 | — |

**8B stage6 (0.7380) 与 8B stage1 (0.7394) 几乎持平 (-0.14pp)** —— 清洁数据 + GPT-5.5 合成在 8B 上**没有 net 增益**。

## 2. Per-task 对比

| Task | 8B stage1 | **8B stage6** | Δ | 14B stage6 |
|---|---|---|---|---|
| Belief | 0.6937 | 0.6866 | -0.71pp | 0.7324 |
| Desire | 0.5917 | 0.5722 | **-1.95pp** ↓ | 0.5833 |
| Emotion | 0.7286 | 0.7214 | -0.72pp | 0.7274 |
| False Belief | 0.8520 | 0.8453 | -0.67pp | 0.8791 |
| **Intention** | 0.7647 | **0.7853** | **+2.06pp** ✓ | 0.8353 |
| **Knowledge** | 0.4792 | 0.4810 | +0.18pp ≈ | 0.5017 |
| Non-literal Comm | 0.7674 | 0.7687 | +0.13pp ≈ | 0.7660 |

**清洁数据在 8B 上的体现**:
- ✅ **Intention +2.06pp** (GPT-5.5 hinting 数据起效)
- ≈ **Knowledge / Non-literal Comm** 持平
- ❌ **Belief / Desire / Emotion** 略降 0.7-2pp

**对比 14B stage6 同样配置**: 14B 的 Knowledge 涨了 +2.42pp, 8B 仅 +0.18pp。**这证明数据红利依赖模型 capacity** —— 8B 不能像 14B 那样真正吸收 GPT-5.5 的 scalar implicature 推理模式。

## 3. Subset500 (3 协议)

| Protocol | 8B stage1 | 8B stage5 | **8B stage6** | 14B stage6 |
|---|---|---|---|---|
| direct | 0.7460 | 0.7340 | **0.7440** | 0.7780 |
| cot | 0.6980 | 0.7380 | **0.7340** | 0.7560 |
| del_tom | 0.7460 | 0.7480 | **0.7500** | **0.7880** |

**8B stage6 best subset500 = del_tom 0.7500** ≈ 8B stage5 best 0.7520 (无突破)。

## 4. 训练动态

**Val (subset500)**:

| step | 8B stage1 | 8B stage5 | **8B stage6** | 14B stage6 |
|---|---|---|---|---|
| 0 | 0.042 | 0.040 | 0.038 | 0.062 |
| 50 | 0.204 | 0.184 | 0.202 | **0.496** |
| 100 | 0.454 | 0.428 | **0.490** (+3.6pp vs s1) | **0.628** |
| 150 | 0.548 | 0.582 | **0.564** (+1.6pp vs s1) | **0.652** |
| 200 | — | 0.530 | **0.594** | **0.662** |

**关键差异 (8B vs 14B 同样配置)**:
- 14B stage6 在 step 50 已 0.496, 比 14B stage1 同期 (0.348) 高 +14.8pp ← 加速明显
- **8B stage6 在 step 50 仅 0.202**, 与 8B stage1 (0.204) 持平 ← 无加速

**Rollout score** 也呈类似差异: 14B stage6 step 100 = 0.96 (saturate), 8B stage6 step 100 仅 ~0.66。

## 5. 关键 insight

### 5.1 数据红利依赖容量

**14B + 清洁数据 → +0.53pp full 5718, Knowledge +2.42pp**
**8B + 清洁数据 → -0.14pp full 5718, Knowledge +0.18pp**

GPT-5.5 合成的 scalar implicature 题需要"语用推理 → 数值推理"的两步链。8B 不能稳定执行此链, 所以即使数据完美, 模型也学不会。

### 5.2 8B 训练上限基本确定

5 个 stage (stage1-3, stage5, stage6) 尝试不同配方:
- max_steps: 200 / 250 / 300 / 500
- KL: on/off
- response_length: 256 / 384
- 数据: 4k / 8k / 8901 / 7259

**最高 direct (full 5718): 8B stage1 0.7394** (最简单的 baseline 配方)

后续配方都 ≈ 0.73 ± 0.01。**8B + RL + 现有数据范式的 ceiling 是 ~0.74**。

### 5.3 生产建议矩阵

| Use case | 推荐模型 | 协议 | 分数 |
|---|---|---|---|
| 轻量部署 / 8B GPU | **qwen3-8b-tom stage1** | direct | 0.7394 |
| 8B + 多协议 | qwen3-8b-tom stage5 | del_tom | 0.7520 (subset) |
| **生产首选** | **qwen3-14b-tom stage6** | **direct/del_tom** | **0.758/0.788** |
| Knowledge 任务 | qwen3-14b-tom stage6 | direct | 0.502 (vs 8B 0.481) |

## 6. 部署

- **HF 模型**: `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage6/` (16 GB)
- **vLLM serve**: `qwen3-tom-serve-8b-stage6`, port 8000, model id `qwen3-8b-tom-stage6`

## 7. 结论

1. **清洁数据 + GPT-5.5 合成的红利在 8B 上不可见**: 模型 capacity 是吸收数据红利的关键
2. **8B + RL ceiling ≈ 0.74**: 5 个 stage 不同配方均收敛于此
3. **要真正逼近 deepseek/GPT-5.5, 必须用 14B 或更大模型**
4. **Stage1 仍是 8B 最佳 direct 配方**, 简单、稳健、可复现

## 8. 产物清单

| 路径 | 内容 |
|---|---|
| `output/eval/8b_stage6_full5718.{json,md}` | 全量 5718 direct |
| `output/eval/8b_stage6_subset500.{json,md}` | subset500 × 3 协议 |
| `output/analysis/curves_stage6_8b.png` | 训练曲线 |
| `output/analysis/errors_8b_stage6.md` | 错题样本 |
| `logs/train_stage6_1x8_20260517_233141.log` | 完整训练日志 (11 MB) |
| Megatron checkpoint (TRAIN) | `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage6-1x8/.../checkpoint-249/` (~107 GB) |
| HF checkpoint (TRAIN) | `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage6/` (16 GB) |

最后更新: 2026-05-18 12:25
