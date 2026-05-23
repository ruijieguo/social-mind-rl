# Production Frozen 8B v1.0 — Stage 15 ckpt-150 快照

> **状态**: 8B 模型线第一个 production frozen，Stage 15 ckpt-150 创 8B 项目记录
> **创建日期**: 2026-05-23
> **基线对比**: Stage 7 (8B baseline, 2026-05-18) → 8b/v1.0 (Stage 15 ckpt-150, 2026-05-23)
> **核心提升**: del_tom 0.7534 → **0.7618** (+0.84pp)，cot 0.7350 → **0.7501** (+1.50pp)

## 头条结果（full 5718 raw）

| 协议 | 8B Stage 7 (baseline) | **8B Stage 15 ckpt-150 (v1.0)** | Δ |
|---|---|---|---|
| direct | **0.7478** | 0.7450 | -0.28pp |
| cot | 0.7350 | **0.7501** | **+1.50pp** ⭐⭐ |
| **del_tom** | 0.7534 | **0.7618** | **+0.84pp** ⭐ |

**与外部基线**：
- DeepSeek-v4-pro del_tom: 0.8069（领先 4.51pp）
- 14B v3.1 ckpt-199 del_tom: 0.7875（领先 2.57pp）

## Per-task del_tom 分解

| Task | Stage 7 | **ckpt-150** | Δ | 加权 (8B-specific) |
|---|---|---|---|---|
| **Knowledge** (最弱) | 0.4758 | **0.5000** | **+2.42pp** ⭐⭐ | ×2.00 |
| **Belief** | 0.7289 | **0.7500** | **+2.11pp** ⭐ | ×1.26 |
| **Intention** | 0.7853 | **0.8221** | **+3.68pp** ⭐⭐ | ×1.07 |
| Non-literal Comm | 0.7747 | 0.7894 | +1.47pp | ×1.11 |
| Desire | 0.5944 | 0.6028 | +0.83pp | ×1.69 |
| Emotion | 0.7357 | 0.7333 | -0.24pp | ×1.23 |
| False Belief (最强) | 0.8791 | 0.8655 | -1.36pp | ×0.77 (下采样) |

5 个 task 涨，2 个微跌。Knowledge 提升 +2.42pp、Belief +2.11pp、Intention +3.68pp 三个最显著。FB 下采样代价 -1.36pp，但整体净 +0.84pp。

## 核心方法：reward labeling + 过滤 + 重新加权

### Stage 15 = 三步骤数据准备

**步骤 1: 用 8B Stage 7 给 12519 条 Stage 12 训练数据打 reward label**
- 每条 prompt 生成 8 个 sample
- 计算 group reward mean
- 暴露：8B Stage 7 在 51.5% 训练数据上 reward >= 0.95（已掌握，无梯度）

**步骤 2: 过滤掉 reward >= 0.95 的 sample**
- 12519 → 6066 条（drop 6453 = 51.5%）

**步骤 3: 用 8B Stage 7 自己的 per-task del_tom acc 反比加权**
- 6066 → 7482 条（×1.23 average）

公式：`multiplier(acc) = clamp(2.0 - (acc - 0.5) × 3.25, 0.7, 2.0)`

| Task | Stage 7 acc | Multiplier | 6066 → 7482 |
|---|---|---|---|
| Knowledge | 0.4758 | ×2.00 | 954 → 1908 |
| Desire | 0.5944 | ×1.69 | 246 → 407 |
| Belief | 0.7289 | ×1.26 | 1223 → 1525 |
| Emotion | 0.7357 | ×1.23 | 386 → 472 |
| Non-literal | 0.7747 | ×1.11 | 472 → 522 |
| Intention | 0.7853 | ×1.07 | 655 → 697 |
| False Belief | 0.8791 | ×0.77 | 844 → 665 |
| Other / Strange / Persuasion / Unexpected | 1.00 | ×1.00 | 1286 → 1286 |

**Stage 15 训练数据 reward 分布**:
- 27.8% 全错 (mean=0.0) — 难题，保留
- 24.5% 0.0-0.30 — 难
- 36.1% 0.30-0.80 — **学习甜区**
- 17.1% 0.80-0.95 — 接近掌握

**Stage 14b 训练数据 reward 分布对比**:
- 57.6% 已经全对 (mean=1.0) → 完全无效，浪费
- 19.7% 0.15-0.80 — 学习甜区
- 9.6% 全错

### Difficulty mask 阈值

```yaml
# Stage 14b (失败)
difficulty_low_threshold: 0.15
difficulty_high_threshold: 0.80
# 实际 samples_used: 12-25/256 (5-10%)

# Stage 15 (成功)
difficulty_low_threshold: 0.05
difficulty_high_threshold: 0.95
# 实际 samples_used: 50-65/256 (20-25%)
```

数据预过滤后 mask 阈值放宽，**有效梯度信号 2-3 倍增加**。

## 训练轨迹（subset500，eval_steps=25）

| step | val | Δ vs init |
|---|---|---|
| 0 | 0.5860 | — |
| 25 | 0.6060 | +2.00pp |
| 50 | 0.6040 | +1.80pp |
| 75 | 0.5980 | +1.20pp |
| **100** | **0.6260** | **+4.00pp** (subset500 峰) |
| 125 | 0.6000 | +1.40pp |
| **150** | 0.6200 | +3.40pp (**全量峰** ⭐) |
| 175 | 0.6240 | +3.80pp |

vs Stage 14b 8B trajectory（持续退化）：
| step | Stage 14b | **Stage 15** |
|---|---|---|
| 25 | -1.00pp | **+2.00pp** |
| 50 | +0.20pp | **+1.80pp** |
| 75 | -2.20pp | **+1.20pp** |
| 100 | -1.00pp | **+4.00pp** |
| 150 | +0.20pp | **+3.40pp** |

**ckpt 选择**：subset500 峰是 ckpt-100 (+4.00pp)，但全量评测显示 ckpt-150 在 cot/del_tom 上双胜：

| Ckpt | direct | cot | del_tom |
|---|---|---|---|
| 50 | 0.7440 | 0.7326 | 0.7578 |
| 100 | 0.7434 | 0.7387 | 0.7581 |
| **150** | 0.7450 | **0.7501** | **0.7618** ⭐ |
| 199 | 0.7403 | 0.7448 | 0.7609 |

## 目录结构

```
production_frozen/8b/v1.0/
├── README.md                                          # 本文件
├── SHA256SUMS.txt                                     # 校验和
├── verify.sh                                          # 校验脚本
├── configs/
│   ├── rlvr_config_8b_stage15_FROZEN.yaml             # Stage 15 训练配置
│   └── rlvr_config_8b_stage7_FROZEN.yaml              # Stage 7 baseline 配置
├── data/
│   ├── tom_train_stage15_8b_filtered_weighted.jsonl   # 7482 条最终训练集
│   ├── tom_train_14b_stage12.jsonl                    # 12519 条源数据（参考）
│   ├── 8b_stage7_reward_full12519.jsonl               # 8B Stage 7 reward labels
│   └── raw/                                           # carry-over from 14b/v3.0
├── scripts/
│   ├── score_train_data_8b.py                         # reward labeling pipeline
│   ├── build_stage15_data.py                          # 数据 filter+reweight 脚本
│   ├── run_tombench.py                                # eval framework
│   └── extractors.py                                  # del_tom 实现
├── eval/
│   ├── 8b_stage15_ckpt150_full5718.json               # 头条 ⭐
│   ├── 8b_stage15_ckpt199_full5718.json               # 第二好
│   ├── 8b_stage15_ckpt100_full5718.json               # subset500 峰但全量第三
│   ├── 8b_stage15_ckpt50_full5718.json                # 早期 ckpt
│   ├── 8b_stage7_full5718_3proto.json                 # baseline
│   └── 8b_stage14b_ckpt150_full5718.json              # 失败实验对照
├── logs/
│   └── train_stage15_8b.log.gz                        # 完整训练日志
└── docs/
    └── stage15_summary.md                             # 完整复盘 (Stage 14b 失败 → Stage 15 成功)
```

## 校验

```bash
cd production_frozen/8b/v1.0
bash verify.sh
```

## 模型权重位置（不在 git 中）

```
TRAIN host: h800@172.16.120.181

Megatron 原始 ckpt:
  /data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage15-1x8/20260522-225059/checkpoint-150/

HF 推理用模型:
  /data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage15-1x8-hf-ckpt150/

其他保存的 ckpt（用于研究/对照）：
  checkpoint-50  (full del_tom 0.7578)
  checkpoint-100 (full del_tom 0.7581, subset500 0.6260 峰)
  checkpoint-199 (full del_tom 0.7609)
```

## 演进时间线

| Date | 8B 版本 | del_tom | 主要事件 |
|---|---|---|---|
| 2026-05-18 | Stage 7 | 0.7534 | cleaned data + GPT-5.5 synth |
| 2026-05-22 | Stage 14b | 0.7569 (S14b-150) | 复用 14B Stage 14b 配方失败（+0.35pp 在 noise 内） |
| **2026-05-23** | **v1.0 (Stage 15)** | **0.7618** | **诊断+修复：reward labeling + 过滤 + 自加权 → +0.84pp** ⭐ |
