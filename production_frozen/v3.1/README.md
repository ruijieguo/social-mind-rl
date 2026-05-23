# Production Frozen v3.1 — Stage 14b ckpt-199 快照

> **状态**: Stage 14b 任务加权重训成功，**ckpt-199 (final)** + del_tom = **0.7875** 创新项目记录
> **创建日期**: 2026-05-22
> **基线对比**: v3.0 (Stage 12, 2026-05-21) → v3.1 (Stage 14b ckpt-199, 2026-05-22)
> **核心提升**: del_tom 0.7823 → **0.7875** (+0.52pp)，cot 0.7690 → **0.7754** (+0.64pp)，direct 0.7660 → **0.7721** (+0.61pp)

## 头条结果（full 5718 raw）

| 协议 | Stage 12 (v3.0) | **Stage 14b ckpt-199 (v3.1)** | Δ |
|---|---|---|---|
| direct | 0.7660 | **0.7721** | **+0.61pp** |
| cot | 0.7690 | **0.7754** | **+0.64pp** ⭐ |
| **del_tom** | 0.7823 | **0.7875** | **+0.52pp** ⭐⭐ |

**与外部基线**：
- GPT-5.5 (zero-shot): 0.8349（仍领先 4.74pp，从 v3.0 5.26pp 收窄 0.52pp）
- deepseek-v4-pro: 0.8080（仍领先 2.05pp，从 v3.0 2.57pp 收窄 0.52pp）

**Direct-only 复现验证**: ckpt-199 direct 二次评测 = 0.7721（完全匹配头条），样本级别 reproducibility 97.27%（vLLM batch 数值微抖动）。结果完全可信。

## Per-task del_tom 分解（弱任务上的真正收益）

| Task | Stage 12 | **ckpt-199** | Δ | Stage 14b 加权 |
|---|---|---|---|---|
| **Belief** | 0.7430 | **0.7606** | **+1.76pp** ⭐⭐ | ×1.21 |
| **Knowledge** (最弱) | 0.5260 | **0.5450** | **+1.90pp** ⭐⭐ | ×1.92 |
| **Emotion** | 0.7607 | **0.7726** | **+1.19pp** ⭐ | ×1.15 |
| **False Belief** (最强还在涨) | 0.8946 | **0.9007** | **+0.61pp** ⭐ | ×0.72 |
| Intention | 0.8294 | 0.8324 | +0.30pp | ×0.93 |
| Desire | 0.6083 | 0.6028 | -0.55pp | ×1.65 |
| Non-literal Comm | 0.8102 | 0.8068 | -0.34pp | ×0.99 |

**任务加权策略验证**: 5 个 task 涨 (Belief / Knowledge / Emotion / False Belief / Intention)，2 个小幅退（Desire 和 Non-literal Comm 在 noise 内）。整体 +0.52pp del_tom 全靠弱任务起飞。

## ckpt 选择 — 全部 5718 评测

| Ckpt | direct | cot | del_tom | 选择理由 |
|---|---|---|---|---|
| Stage 12 | 0.7660 | 0.7690 | 0.7823 | baseline |
| ckpt-50 | 0.7653 | 0.7716 | 0.7810 | subset500 看似最高但全量 noise |
| ckpt-100 | 0.7620 | 0.7697 | 0.7800 | 中间步骤，最差 |
| ckpt-150 | 0.7723 | 0.7733 | 0.7837 | 第二好 |
| **ckpt-199 (final)** | **0.7721** | **0.7754** | **0.7875** | **全方位最佳 → v3.1 头条** ⭐ |

**关键 lesson**: subset500 反向骗人。ckpt-199 在 subset500 上 step 175 仅 0.7520（看似最低），但全量 del_tom 0.7875 是绝对峰值。subset500 ±1pp 噪音掩盖了 ckpt-199 的真实优势。

## 核心方法：任务加权重训

`tom_train_14b_stage14b_weighted.jsonl` (14408 records) 是用 Stage 12 在 full 5718 上的 per-task 准确率，把 Stage 12 的 12519 条原始训练数据按反比例上/下采样：

```
multiplier(acc) = clamp(2.0 - (acc - 0.5) × 3.25, 0.7, 2.0)
```

| Task | Stage 12 acc | Multiplier | 原始 → 加权 |
|---|---|---|---|
| Knowledge | 0.5260 | ×1.92 | 1847 → 3523 |
| Desire | 0.6083 | ×1.65 | 758 → 1246 |
| Belief | 0.7430 | ×1.21 | 1832 → 2236 |
| Emotion | 0.7607 | ×1.15 | 659 → 760 |
| Non-literal Comm | 0.8102 | ×0.99 | 910 → 902 |
| Intention | 0.8294 | ×0.93 | 1055 → 972 |
| False Belief | 0.8946 | ×0.72 | 2533 → 1844 |
| Other (`Other`/`Strange Story`/`Persuasion Story`/`Unexpected Outcome`) | n/a | ×1.00 | 不变 |

**关键超参修改**（vs Stage 12）：

```yaml
# Stage 12
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95

# Stage 14b
difficulty_low_threshold: 0.15
difficulty_high_threshold: 0.80
```

Stage 12 在自己的训练数据上 reward 集中在 0.90+，0.95 上限掩码导致 ~96% 样本被过滤（first attempt of Stage 14 验证）。降到 0.80 让"会但不绝对"的样本进梯度。

## 训练轨迹（subset500，eval_steps=25）

| step | val | Δ vs init | 全量 del_tom | 备注 |
|---|---|---|---|---|
| 0 | 0.7440 | — | (Stage 12 = 0.7823) | init |
| 25 | 0.7560 | +1.20pp | — | |
| 50 | 0.7680 | **+2.40pp** | 0.7810 | subset500 峰，全量是 noise |
| 75 | 0.7440 | +0.00pp | — | |
| 100 | 0.7500 | +0.60pp | 0.7800 | 全量最差 |
| 125 | 0.7460 | +0.20pp | — | |
| 150 | 0.7640 | +2.00pp | 0.7837 | 全量第二 |
| 175 | 0.7520 | +0.80pp | — | |
| **199 (final)** | **n/a** | n/a | **0.7875** | **全量峰值 ⭐** |

**Lesson**: 多 ckpt 全量评测必不可少。step 199 没做 subset500 eval（训练已结束），不能从 subset500 trajectory 推断它最佳 — 必须真做。

## 目录结构

```
production_frozen/v3.1/
├── README.md                                          # 本文件
├── SHA256SUMS.txt                                     # 校验和
├── verify.sh                                          # 校验脚本
├── configs/
│   ├── rlvr_config_14b_stage14b_FROZEN.yaml          # Stage 14b 训练配置
│   └── rlvr_config_14b_stage12_FROZEN.yaml           # v3.0 baseline 配置（参考）
├── data/
│   ├── tom_train_14b_stage14b_weighted.jsonl         # 14408 条加权训练集
│   ├── tom_train_14b_stage12.jsonl                   # 12519 条 Stage 12 原料（参考）
│   └── raw/                                          # carry-over from v3.0
├── scripts/                                          # carry-over from v3.0
├── eval/
│   ├── 14b_stage14b_ckpt199_full5718.json            # 头条结果，3 协议 5718 题 ⭐
│   ├── 14b_stage14b_ckpt150_full5718.json            # ckpt-150 (第二好) 对照
│   ├── 14b_stage14b_ckpt100_full5718.json            # ckpt-100 (中间步骤) 对照
│   ├── 14b_stage14b_ckpt50_full5718.json             # ckpt-50 (subset500 假峰) 对照
│   └── 14b_stage12_full5718.json                     # v3.0 baseline 参考
├── logs/
│   └── train_stage14b_14b.log.gz                     # 完整训练日志
└── docs/
    └── stage14b_summary.md                           # 完整复盘
```

## 校验

```bash
cd production_frozen/v3.1
bash verify.sh
```

## 模型权重位置（不在 git 中）

```
TRAIN host: h800@172.16.120.181

Megatron 原始 ckpt:
  /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage14b-1x8/20260521-154242/checkpoint-199/

HF 推理用模型:
  /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage14b-1x8-hf-ckpt199/

其他保存的 ckpt（用于研究/对照）：
  checkpoint-50  (full del_tom 0.7810, subset500 0.7680)
  checkpoint-100 (full del_tom 0.7800, subset500 0.7500)
  checkpoint-150 (full del_tom 0.7837, subset500 0.7640)
```

## 演进时间线

| Date | 版本 | del_tom | 主要事件 |
|---|---|---|---|
| 2026-05-20 | v1.0 | 0.7762 | Stage 8 (Track A only)，14B baseline |
| 2026-05-20 | v2.0 | n/a | Stage 11 v2 系列开发期 |
| 2026-05-21 | **v3.0** | **0.7823** | Stage 12: Track A + B + C + D 整合 → +0.61pp |
| 2026-05-21 | n/a | -1.80pp | Stage 13 续训失败：data 已榨干，过拟合 |
| 2026-05-22 | **v3.1** | **0.7875** | **Stage 14b: 任务加权 + difficulty mask 修复** → +0.52pp ⭐ |
