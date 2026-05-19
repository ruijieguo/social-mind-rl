# Stage 7 8B 报告（中文版）：Phase A 数据扩展（9559 records）

> 训练: 2026-05-19 03:31 → 07:40 UTC；250 步, 9559 训练数据
> 评测: 2026-05-19 15:20（full 5718 + clean 4551 + subset500 × 3 协议）
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage7-1x8/20260518-192042/checkpoint-249/`

## 1. 关键结果 — 8B 首次突破 0.74 ceiling

### Full 5718 direct
| 8B 模型 | Direct | Δ vs s1 baseline |
|---|---|---|
| qwen3-8b-nt | 0.7009 | — |
| qwen3-8b-tom stage1 (旧最高) | 0.7394 | +3.85pp |
| qwen3-8b-tom stage2 | 0.7263 | -1.31 |
| qwen3-8b-tom stage3 | 0.7302 | -0.92 |
| qwen3-8b-tom stage5 | 0.7305 | -0.89 |
| qwen3-8b-tom stage6 | 0.7380 | -0.14 |
| **qwen3-8b-tom stage7** | **0.7419** | **+0.25pp** ⭐ 首次破 0.74 ceiling |
| qwen3-14b-tom stage8 | 0.7594 | — |
| deepseek-v4-pro | 0.8080 | — |
| GPT-5.5 | 0.8349 | — |

### Clean Eval 4551 direct
| 模型 | Direct | Δ vs raw |
|---|---|---|
| **qwen3-8b-tom stage7** | **0.8321** | +9.0pp |
| qwen3-14b-tom stage8 | 0.8449 | +8.6pp |
| deepseek-v4-pro | 0.9013 | +9.3pp |
| GPT-5.5 | 0.9343 | +9.9pp |

### Subset500 (3 协议) — 8B cot 历史新高
| Protocol | s1 | s5 | s6 | **s7** |
|---|---|---|---|---|
| direct | 0.7460 | 0.7340 | 0.7440 | **0.7440** |
| **cot** | 0.6980 | 0.7380 | 0.7340 | **0.7460** ⭐ |
| del_tom | 0.7460 | 0.7480 | 0.7500 | 0.7480 |

## 2. 数据 — Phase A 完整组合 (9559 records)

| 来源 | 数量 |
|---|---|
| base (post-GPT-5.5-audit cleanup) | 7259 |
| Phase A.1: social_norm (EN+ZH) | 400 |
| Phase A.1: factual_detail (EN+ZH) | 300 |
| Phase A.1: intention_attribution (EN+ZH) | 400 |
| Phase A.1: indirect_speech_act (EN+ZH) | 400 |
| Phase A.2: belief_zh | 300 |
| Phase A.2: knowledge_zh | 250 |
| Phase A.2: desire_zh | 250 |
| **TOTAL** | **9559** |

Phase A.1 设计长故事 (8-12 句 GPT-5.5 风格)，**这是 stage7 14B 失败的原因**（风格不匹配）。但 **8B 受益**：模型容量小，对故事风格更宽容，Phase A 的高质量数据本身带来红利。

## 3. Per-task 详细对比

**Full 5718 (vs s6 与 s1)**:
| Task | s1 | s6 | **s7** | Δs7-s6 | Δs7-s1 |
|---|---|---|---|---|---|
| **Belief** | 0.694 | 0.687 | **0.701** | **+1.4** ⭐ | +0.7 |
| **Desire** | 0.592 | 0.572 | **0.608** | **+3.6** ⭐ | +1.7 |
| Emotion | 0.729 | 0.721 | 0.720 | -0.1 | -0.8 |
| **False Belief** | 0.852 | 0.845 | **0.857** | **+1.1** ⭐ | +0.5 |
| Intention | 0.765 | 0.785 | 0.760 | -2.5 | -0.4 |
| Knowledge | 0.479 | 0.481 | 0.478 | -0.3 | -0.2 |
| **Non-literal Comm** | 0.767 | 0.769 | **0.774** | +0.5 | +0.7 |

**关键胜利**：
- **Desire +3.6pp**: Phase A.2 中文 desire (250) + Phase A.1 intention (400) 帮助
- **Belief +1.4pp**: Phase A.1 factual_detail (300) 训练细读
- **False Belief +1.1pp**: 间接受益 + Belief 改善
- **Non-literal +0.5pp**: Phase A.1 social_norm 起效

**仍有挑战**：
- **Intention -2.5pp vs s6**: 数据稀释（7259→9559 但 max_steps=250 不变，每条样本见次数 5x vs s6 的 7x）
- **Knowledge 持平**: 没有 scalar 数据帮助（phase_a 不含 Knowledge 专项）
- **Emotion -0.1pp**: 略受 Intention 拖累

## 4. 训练动态对比

**Val (subset500)**:
| step | 8B s1 | 8B s6 | **8B s7** |
|---|---|---|---|
| 0 | 0.042 | 0.038 | 0.036 |
| **50** | 0.204 | 0.202 | **0.260** ⭐ +5.6pp |
| 100 | 0.454 | 0.490 | 0.460 (-3.0) |
| 150 | 0.548 | 0.564 | 0.574 (+1.0) |
| 200 | — | 0.594 | 0.600 (+0.6) |
| 250 (final) | — | (final) | (final) |

**8B vs 14B 不同表现**:
- **14B step 50**: s7 0.516 = s6 0.516（小幅）
- **8B step 50**: s7 0.260 > s6 0.202 (+5.6pp 大幅)

**8B 在前期更快受益于 Phase A 高质量数据**，但中期 (step 100) 出现风格漂移（-3.0pp vs s6），后期再 stabilize。

## 5. 关键 insight

### 5.1 8B 终于突破 stage1 ceiling
连续 5 个 stage（stage2/3/5/6）尝试不同配方都没破 stage1 0.7394。**Phase A 的 2300 条新数据让 8B +0.25pp 破 ceiling**，证明 stage6 时 8B "持平 stage1" 不是模型上限，而是数据量上限。

### 5.2 Phase A 在 8B 上有效，在 14B 上无效
| 模型 | s6 | Phase A 加入 | Δ |
|---|---|---|---|
| 8B | 0.7380 | **0.7419** | **+0.25pp** ✓ |
| 14B | 0.7580 | 0.7539 (s7) | -0.41pp ✗ |

**完全相反的结果**。可能解释：
1. 8B 模型小，容量稀缺，**任何质量提升都有边际收益**
2. 14B 容量大，已经从 stage6 数据榨取了所有信号，新数据的风格不匹配反成噪声
3. 14B 在 val (含合成数据) 上学得好，但 full eval（ToMBench 风格）上反退

### 5.3 8B/14B 推荐数据策略不同
**8B**: 用 Phase A 数据 (stage7) 是最佳，breaks 0.74 ceiling
**14B**: 用 Phase C 风格匹配数据 (stage8) 是最佳，达 0.7594

## 6. 实用决策

**8B 生产部署**: **stage7 8B**（取代 stage1）
- direct: 0.7419（+0.25pp vs s1, 史上最高）
- cot: 0.7460（subset500 best, 8B cot 历史最高）
- HF model: `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage7/` (16 GB)
- vLLM serve: `qwen3-tom-serve-8b-stage7` (port 8000, model id `qwen3-8b-tom-stage7`)

**14B 生产部署**: **stage8 14B** (见 docs/stage8_report.md)

## 7. 项目累计成绩

| Model | full direct | clean direct | subset500 best |
|---|---|---|---|
| qwen3-8b-nt | 0.7009 | — | — |
| qwen3-8b-tom stage1 | 0.7394 | — | 0.7460 (direct) |
| **qwen3-8b-tom stage7** | **0.7419** ⭐ | **0.8321** | **0.7480** (del_tom) |
| qwen3-14b-nt | 0.7338 | — | — |
| qwen3-14b-tom stage8 | 0.7594 | 0.8449 | 0.7920 (del_tom) |
| deepseek-v4-pro | 0.8080 | 0.9013 | 0.7880 (direct) |
| GPT-5.5 | 0.8349 | 0.9343 | — |

8B 距 14B stage8: -1.75pp (raw) / -1.28pp (clean)
8B 距 deepseek: -6.61pp (raw) / -6.92pp (clean)
8B 距 GPT-5.5: -9.30pp (raw) / -10.22pp (clean)

## 8. 产物清单

| 路径 | 内容 |
|---|---|
| `output/eval/8b_stage7_full5718.{json,md}` | full 5718 direct |
| `output/eval/8b_stage7_clean_eval.{json,md}` | clean 4551 |
| `output/eval/8b_stage7_subset500.{json,md}` | subset500 × 3 |
| `output/analysis/curves_stage7_8b.png` | 训练曲线 |
| `output/analysis/errors_8b_stage7.md` | 错题样本 |
| `logs/train_stage7_1x8_20260518_192021.log` | 训练日志 (12 MB) |
| Megatron ckpt | `qwen3-8B-tombench-rlvr-stage7-1x8/.../checkpoint-249/` (~107 GB) |
| HF ckpt | `qwen3-8B-tom-hf-stage7/` (16 GB) |

最后更新: 2026-05-19 15:30
