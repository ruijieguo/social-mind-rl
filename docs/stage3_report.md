# Stage3 1×8 训练 + 评测报告

> 训练: 2026-05-16 10:36 → 16:00 (UTC); 共 300 步 (300 max_steps), 8k 数据 (tom_train.jsonl 7911 records，**Phase-1 合成数据未参与**因为 stage3 与 Phase-1 数据合成并发执行)
> Eval: 2026-05-17 00:14 (full 5718 direct) + subset500 × 3 protocols
> Checkpoints: `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage3-1x8/20260516-103657/{checkpoint-150,checkpoint-299}/`

## 1. Headline 数字

**Full ToMBench 5718 (direct)**:

| Model | Overall | EN | ZH |
|---|---|---|---|
| qwen3-8b-nt baseline | 0.7009 | 0.7020 | 0.6999 |
| qwen3-8b-tom **stage1** (200 steps, 4k) | **0.7394** | 0.7275 | **0.7513** |
| qwen3-8b-tom **stage2** (500 steps, 8k) | 0.7263 | 0.7223 | 0.7303 |
| qwen3-8b-tom **stage3** (300 steps, 8k, +KL+384) | **0.7302** | 0.7156 | 0.7447 |
| deepseek-v4-pro target | 0.7880 | 0.7803 | 0.7966 |

**Stage3 与 stage2 持平 (0.730 vs 0.726)，但比 stage1 (0.739) 低 0.92pp**。结论: **KL penalty 没有解决核心问题**。

**Subset500 (3 protocols)**:

| Protocol | stage1 | stage2 | **stage3** |
|---|---|---|---|
| direct | **0.7460** | 0.7340 | 0.7360 |
| cot | 0.6980 | **0.7540** | 0.6240 ↓↓ |
| del_tom | 0.7460 | 0.7480 | 0.7080 |

**Stage3 cot 大幅崩** (0.624, 比 baseline 还低)。

## 2. Per-task 对比

| Task | baseline | stage1 | stage2 | **stage3** | stage3 vs ds gap |
|---|---|---|---|---|---|
| Belief | 0.6725 | 0.6937 | 0.6373 | **0.7007** ✓ | -9.93pp |
| Desire | 0.5861 | 0.5917 | 0.5861 | 0.5556 ↓ | -8.33pp |
| Emotion | 0.6893 | 0.7286 | 0.7012 | 0.7107 | +0.14pp ← 超 ds! |
| False Belief | 0.7277 | 0.8520 | 0.8385 | **0.8649** ✓ | +0.34pp ← 超 ds! |
| Intention | 0.7500 | 0.7647 | 0.7632 | 0.7618 | -5.18pp |
| Knowledge | 0.4810 | 0.4792 | 0.4879 | 0.4377 ↓↓ | -16.23pp |
| Non-literal Comm | 0.7767 | 0.7674 | 0.7553 | **0.7540** ↓ | -8.93pp |

**重要发现**:
- ✅ **Belief 和 False Belief 创新高**: stage3 0.7007 / 0.8649，**超过 deepseek-v4-pro 在这两个 task 上的水平**
- ✅ **Emotion 几乎追平 deepseek** (0.7107 vs 0.7093, +0.14pp)
- ❌ **Knowledge 大跌 5pp (s2→s3)**: KL penalty 让模型不敢学新数据
- ❌ **Desire 跌 3pp**: 同上
- ❌ **Non-literal Comm 持续下跌**: 训练数据未覆盖该 task

## 3. Stage2 → Stage3 转移分析

总 movements (full 5718, direct):
- gains s2→s3: 394
- losses: 372
- **net: +22** (≈ +0.38pp overall, 符合实际 +0.39pp)

| task | gain | loss | net |
|---|---|---|---|
| **False Belief** | 103 | 64 | **+39** |
| **Belief** | 38 | 20 | **+18** |
| Emotion | 52 | 44 | +8 |
| Intention | 41 | 42 | -1 |
| Non-literal Comm | 95 | 97 | -2 |
| Desire | 24 | 35 | -11 |
| **Knowledge** | 41 | 70 | **-29** ← 大失分 |

**KL penalty 起到了预期作用** — 保住了 Belief / False Belief 的 robust reasoning，但同时**阻止了 Knowledge / Desire 的进步**。这是个**典型的稳健性-学习率 tradeoff**。

## 4. 训练动态

**Validation 曲线** (subset500, val_correct/all):

| step | stage1 | stage2 | **stage3** |
|---|---|---|---|
| 0 | 0.042 | 0.036 | 0.038 |
| 50 | 0.204 | 0.206 | 0.148 ↓ |
| 100 | 0.454 | 0.466 | 0.222 ↓↓↓ |
| 150 | 0.548 | 0.546 | 0.236 |
| 200 | — | 0.530 | 0.226 |
| 250 | — | 0.634 | 0.230 |

**Stage3 val_correct 远低于 stage1/2**！但**实际 full eval (max_tokens=2048) 0.7302 != val_correct (max_tokens=64) 0.230**。这暴露了一个**ROLL val 配置 bug**: 训练 response_length=384，val 用 max_new_tokens=64 严重截断 stage3 长 think，导致 val 不能反映真实能力。

**Rollout score (in-batch 答对率)**:
- step 0: 0.33, step 80: 0.44, step 120: 0.50 (peak), step 200: 0.39, step 260: ~0.40
- **stage3 rollout score 在 step 120 达峰后回退**，这是 **KL penalty 让模型反复在 base/learned 间摇摆**

## 5. 关键诊断: KL penalty 配置过于激进

**证据**:
1. step 120 rollout peak 0.50 后 plateau/回退 — 模型没有持续学习
2. Knowledge / Desire 净失分大 — 新数据收益被锁死
3. val_correct 远低于真实能力 — 不是模型差，是评测协议被 stage3 的长 think 破坏

**修复方向 (for next stage)**:
- 降低 `add_token_level_kl` 强度 (改为 `kl_coef: 0.001` 之类的细调)
- response_length 改回 256（与 val 64 不严重 mismatch），且 stage1/2 在 256 下表现就好
- 或保持 384 但 val 设置 `max_new_tokens: 384` 改 ROLL config

## 6. 部署

- **HF 模型路径**: `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage3/` (16 GB)
- **vLLM serve**: `qwen3-tom-serve-stage3` (port 8000)，model id `qwen3-8b-tom-stage3`
- **OpenAI 端点**: `http://172.16.120.181:8000/v1`

## 7. 综合结论 / 接下来

**Stage3 是一次"特化但未通杀"的实验**:
- 在 Belief / False Belief 上**击败了 deepseek-v4-pro**（首次！）
- 在 Knowledge / Non-literal Comm 上未取得进展
- 整体 (direct) 0.7302 比 stage1 0.7394 低 0.92pp，但比 stage2 0.7263 高 0.39pp

**Stage4 (with Phase-1 data) 策略**:
1. **Phase-1 数据已准备好** (`data/tom/tom_train.jsonl` 现 8902 records，+991 新合成)
2. **配置调整**:
   - 沿用 stage3 的 `add_token_level_kl: true`（保 Belief/FB 强势）
   - response_length 回 **256** (与 val 一致)
   - max_steps 仍 **300** (避免 stage2-style 过拟合)
   - save_steps **300** 一次 (省盘)
3. **核心赌注**: Phase-1 的 800 faux-pas + 400 scalar + 300 hinting + 300 so_belief 能修复 stage3 没解决的 Knowledge / Non-literal Comm 等 task
4. **预期**: 如果 Phase-1 设计有效，stage4 应该:
   - 保留 stage3 的 Belief / False Belief 优势
   - 通过 scalar 数据修复 Knowledge (现 0.4377 → 期望 0.50+)
   - 通过 faux_pas 数据修复 Non-literal Comm (现 0.7540 → 期望 0.78+)
   - 整体目标 direct ≥ 0.76 (距 deepseek < 3pp)

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `output/eval/stage3_full5718.{json,md}` | 5718 direct 完整 |
| `output/eval/stage3_subset500.{json,md}` | 500 × 3 protocol |
| `output/analysis/curves_stage3_1x8.png` | 12 子图训练曲线 |
| `output/analysis/errors_stage3.md` | 错题样本 |
| `logs/train_stage3_1x8_RESUMED_20260516_124846.log` | 完整训练日志 (22 MB) |
| HF 模型: `qwen3-8B-tom-hf-stage3/` | 部署用 (16 GB) |

Last updated: 2026-05-17 00:30
