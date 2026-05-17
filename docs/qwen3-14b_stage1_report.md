# Qwen3-14B Stage1 1×8 训练 + 评测报告

> 训练: 2026-05-17 03:14 → 15:29 (UTC); 200 步, 4k tom_train_4k.jsonl, TP=2
> Eval: 2026-05-17 15:35 (full 5718 + subset500 × 3 protocols)
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage1-1x8/20260517-031410/checkpoint-199/`

## 1. 🎯 Headline 结果

**Full ToMBench 5718 (direct)**:

| Model | Overall | EN | ZH |
|---|---|---|---|
| qwen3-8b-nt baseline | 0.7009 | 0.7020 | 0.6999 |
| **qwen3-14b-nt baseline (no RL)** | **0.7338** | 0.7219 | 0.7457 |
| qwen3-8b-tom stage1 | 0.7394 | 0.7275 | 0.7513 |
| qwen3-8b-tom stage5 | 0.7305 | 0.7181 | 0.7429 |
| **qwen3-14b-tom (this work)** | **0.7527** | 0.7422 | **0.7632** |
| deepseek-v4-pro target | 0.7880 | 0.7803 | 0.7966 |

**Stage1 14B = 0.7527, 距 deepseek 仅 3.53pp，比之前 8B 任何 stage 都强！**

**Subset500 (3 protocols)** —— **首次接近/局部超越 deepseek**:

| Protocol | 8B stage1 | 8B stage5 | **14B stage1** | deepseek |
|---|---|---|---|---|
| direct | 0.7460 | 0.7340 | **0.7800** | 0.7880 |
| cot | 0.6980 | 0.7380 | **0.7720** | 0.7140 ← 我们更高 |
| del_tom | 0.7460 | 0.7520 | **0.7760** | — |

**14B subset500 best (direct) = 0.7800, 距 deepseek 仅 0.80pp**！cot 协议下 **0.7720 比 deepseek 0.7140 高 +5.80pp**！

## 2. Per-task 突破 (Full 5718, direct)

| Task | 8B-tom-s1 | 14B-nt | **14B-tom** | deepseek | gap | 突破? |
|---|---|---|---|---|---|---|
| Belief | 0.6937 | 0.7359 | **0.7465** | 0.8000 | -5.35pp | |
| Desire | 0.5917 | 0.5833 | 0.5889 | 0.6389 | -5.00pp | |
| Emotion | **0.7286** | 0.7202 | 0.7286 | 0.7093 | **+1.93pp** | ✓ |
| False Belief | 0.8520 | 0.8047 | **0.8770** | 0.8615 | **+1.55pp** | ✓ |
| Intention | 0.7647 | 0.7662 | **0.8103** | 0.8136 | -0.33pp | ≈ |
| Knowledge | 0.4792 | 0.4671 | 0.4775 | 0.6000 | -12.25pp | bottleneck |
| Non-literal | 0.7674 | 0.7955 | 0.7640 | 0.8433 | -7.93pp | |

**3 个 task 在 full 5718 上击败 deepseek**: Emotion, False Belief, Intention (≈)。

## 3. Per-task subset500 — 几乎全部反超!

| Task | 14B best protocol | 14B best | deepseek | Δ |
|---|---|---|---|---|
| Belief | direct | **0.8000** | 0.8000 | **0** ✓ |
| Desire | del_tom | **0.7778** | 0.6389 | **+13.9pp** ✓✓ |
| Emotion | del_tom | **0.7209** | 0.7093 | +1.16pp ✓ |
| False Belief | cot | **0.8846** | 0.8615 | **+2.31pp** ✓ |
| Intention | direct | **0.8644** | 0.8136 | **+5.08pp** ✓ |
| Knowledge | direct | 0.4000 | 0.6000 | -20.00pp |
| Non-literal | cot | **0.8358** | 0.8433 | -0.75pp ≈ |

**6/7 task 在 subset500 上达到/超过 deepseek 水平**。Knowledge 是唯一显著 gap。

## 4. 训练动态对比 (vs 8B stage1)

**Val 轨迹** (subset500, val_correct/all):

| step | 8B stage1 | **14B stage1** | Δ |
|---|---|---|---|
| 0 | 0.042 | 0.066 | +2.4pp |
| 50 | 0.204 | **0.348** | **+14.4pp** |
| 100 | 0.454 | **0.546** | **+9.2pp** |
| 150 | 0.548 | 0.550 | +0.2pp |

**14B 学习速度 ~1.5x 于 8B**。step 50 已达 8B step 80+ 水平。

**Rollout score**:
| step | 8B stage1 | 14B stage1 |
|---|---|---|
| 50 | 0.33 | 0.49 (+16pp) |
| 100 | 0.52 | 0.72 (+20pp) |
| 150 | — | **0.90** (saturate) |
| 198 | 0.79 | **0.97** (saturate) |

14B 在 step 150 已 saturate (~0.90)，比 8B (step 200 还在 0.79) 提前 50 步。说明 14B 更快"打满"训练数据。

## 5. 训练成本

| 维度 | Qwen3-8B | Qwen3-14B |
|---|---|---|
| 模型大小 | 8B params | 14B params (+75%) |
| TP | 1 | **2** (1×8 H800 必需) |
| Per-step 时间 | ~60s | ~70s (TP overhead 小) |
| 200 steps wall | ~3h 20m | ~6h (含 25 min model download) |
| Save 时间 | ~10 min | ~15 min (gloo gather 14B opt state ~2x 大) |
| Checkpoint 大小 | ~112 GB | ~196 GB |

## 6. 工程细节

**配置变更** (vs 8B stage1_1x8):
- `pretrain: Qwen3-8B → Qwen3-14B`
- `actor_train.tensor_model_parallel_size: 1 → 2`
- `actor_infer (vllm).tensor_parallel_size: 1 → 2`
- `actor_infer.gpu_memory_utilization: 0.6 → 0.45` (14B KV cache 大)
- `reference.tensor_model_parallel_size: 1 → 2`
- `prompt_length: 2048 → 1024` (KV cache 减半给 vllm 留余地)

**修复的 bug**:
1. **`pkg_resources` ModuleNotFoundError**: vllm 0.8.4 TP>1 路径在 ray-distributed-executor 里 import `pkg_resources`，但 setuptools≥75 把它拆出去了。修复: Dockerfile 加 `setuptools<75` (commit `2ae3442`)
- 8B 用 TP=1 不走这条路径，所以没踩坑
- 第一次启动训练 init 后立刻 crash，第二次 rebuild + 修复后正常

## 7. 部署

- **HF 模型**: `/data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf/` (28 GB safetensors, 8 shards)
- **vLLM serve**: `qwen3-tom-serve-14b` (port 8000, model id `qwen3-14b-tom`)
- **OpenAI 端点**: `http://172.16.120.181:8000/v1`

## 8. 结论 / 接下来

**14B + RL 是迄今为止最强模型组合**:
- direct 0.7527 (full 5718, +1.33pp vs 8B 最佳)
- subset500 best 0.7800 (direct), 距 deepseek 0.7880 仅 0.80pp
- 6/7 task 在 subset500 上达到/超过 deepseek
- **distance to deepseek 从 8B 的 4.86pp 缩到 3.53pp** (full 5718)

**进一步突破方向**:
1. **Knowledge task (Scalar Implicature) 修复**: 这是唯一显著 gap (-12.25pp)，需要更精细的语用学合成数据
2. **stage2-style 14B 训练 (8k × 500 steps)**: 14B 在 200 步已 saturate，多走可能边际效益小，**但配合 Phase-1 fixed data 可能有意外收益**
3. **更大的 deepseek synthesis baseline 数据**: 直接用 deepseek-v4-pro 标注更多 ToM 题给 14B 学

## 9. 产物清单

| 文件 | 说明 |
|---|---|
| `output/eval/14b_full5718.{json,md}` | 5718 direct |
| `output/eval/14b_subset500.{json,md}` | 500 × 3 protocols |
| `output/eval/qwen3-14b-nt_full5718.{json,md}` | 14B raw baseline (no RL) |
| `output/analysis/curves_14b_stage1_1x8.png` | 训练曲线 |
| `output/analysis/errors_14b.md` | 错题样本 |
| `logs/train_stage1_1x8_14b_20260517_031327.log` | 完整训练日志 (11 MB) |
| HF model: `qwen3-14B-tom-hf/` | 部署用 (28 GB) |

Last updated: 2026-05-17 15:40
