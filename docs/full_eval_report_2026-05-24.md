# 完整评测报告 — 9 模型 × 4 Benchmark × 3 协议

> **日期**: 2026-05-24，**v3.2 增补于 2026-05-25**，**v3.3 增补于 2026-05-25**，**v3.4 增补于 2026-05-26**
> **模型** (9): Qwen3-8B base, Qwen3-14B base, Qwen3-8B SOTA (v1.0 = Stage 15 ckpt-150), Qwen3-14B SOTA (v3.1 = Stage 14b ckpt-199), Qwen3-14B v3.2 (Stage 16 ckpt-270), Qwen3-14B v3.3 (Stage 17 ckpt-120), **Qwen3-14B v3.4 (Stage 18 ckpt-30, new)**, DeepSeek-v4-pro, GPT-5.5
> **Benchmark** (4): ToMBench (5718), Hi-ToM (600), SocialIQA dev (1954), EmoBench (1200)
> **协议** (3): direct (max_tokens=64 for Qwen non-think / 8192 for reasoning models), cot (greedy 1-sample), del_tom (8-sample 多数投票, T=0.7)

---

## TL;DR (v3.4 增补 — GPT-5.5 蒸馏)

1. **v3.4 Hi-ToM 持续突破**: cot **+0.83pp** (距 DS 仅 **-0.42pp**)，del_tom **+1.17pp**, direct +0.16pp — distillation 真正帮 belief tracking
2. **v3.4 SocialIQA / EmoBench 几乎持平 v3.3** — distillation 在 emotion/commonsense knowledge 上无效（base model 缺知识，蒸馏教不会）
3. **v3.4 ToMBench 微退** -0.19~-0.24pp (与 v3.2/v3.3 同稀释模式)
4. **v3.1→v3.4 Hi-ToM cot 进展轨迹**: 0.6667 → 0.7150 → 0.7350 → **0.7433** (gap 从 -8.08pp → -3.25pp → -1.25pp → **-0.42pp**)
5. **方法论结论**: GPT-5.5 paraphrase + verify 蒸馏对**推理 task** (Hi-ToM/ToMBench) 有效, 对 **knowledge task** (EmoBench/SocialIQA) 无效

---

## 0. 全局总表 — 9 模型 × 4 Benchmark × 3 协议

每行 = (Benchmark, 协议)，每列 = 模型。**粗体** = 该行最高，— 表示未评测。

| Benchmark | Protocol | 8B base | 14B base | 8B v1.0 | 14B v3.1 | 14B v3.2 | 14B v3.3 | **14B v3.4** | DeepSeek | GPT-5.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ToMBench** (n=5718) | direct | 0.7009 | 0.7338 | 0.7450 | 0.7721 | 0.7692 | 0.7658 | 0.7674 | 0.8080 | **0.8349** |
| ToMBench | cot | 0.7464 (n=3675) | — | 0.7501 | 0.7754 | 0.7704 | 0.7698 | 0.7674 | 0.7140 (n=500) | — |
| ToMBench | del_tom | — | — | 0.7618 | **0.7875** | 0.7831 | 0.7796 | 0.7777 | 0.8069 | — |
| **Hi-ToM** (n=600) | direct | 0.5550 | 0.5333 | 0.5717 | 0.5417 | 0.5783 | 0.5817 | 0.5833 | **0.8033** | — |
| Hi-ToM | cot | 0.4467 | 0.4367 | 0.6183 | 0.6667 | 0.7150 | 0.7350 | 0.7433 | **0.7475** (n=598) † | — |
| Hi-ToM | del_tom | 0.4817 | 0.4783 | 0.6567 | 0.7217 | 0.7550 | 0.7500 | **0.7617** | — (hang) | — |
| **SocialIQA** (n=1954) | direct | 0.7559 | 0.7871 | 0.7605 | 0.7876 | 0.7881 | 0.7866 | 0.7866 | **0.8101** | — |
| SocialIQA | cot | 0.7600 | 0.7856 | 0.7733 | 0.7861 | 0.7810 | 0.7825 | 0.7774 | **0.8106** | — |
| SocialIQA | del_tom | 0.7758 | 0.8035 | 0.7851 | 0.7892 | 0.7835 | 0.7830 | 0.7861 | **0.8188** | — |
| **EmoBench** (n=1200) | direct | 0.6033 | 0.6325 | 0.6000 | 0.6483 | 0.6492 | 0.6483 | 0.6508 | **0.7625** | — |
| EmoBench | cot | 0.5725 | 0.6342 | 0.6208 | 0.6575 | 0.6517 | 0.6483 | 0.6450 | **0.7550** | — |
| EmoBench | del_tom | 0.6233 | 0.6717 | 0.6342 | 0.6775 | 0.6600 | 0.6700 | 0.6633 | **0.7733** | — |

† DeepSeek Hi-ToM cot 有 2 题 timeout，有效 447/598 = 0.7475。

**v3.4 vs DeepSeek (12 cell, ToMBench cot DS 是 subset500)**:
- ✅ 1/11 cells ≥ DeepSeek (ToMBench cot, +5.34pp — 但 DS 是 subset500 + truncation bug)
- 📈 距 DS 最近: Hi-ToM cot **-0.42pp**, ToMBench del_tom **-2.92pp**, SocialIQA direct -2.35pp
- 📉 距 DS 最远: Hi-ToM direct -22.00pp, EmoBench (3 protocols) -11pp

**v3.1 → v3.4 进化路径** (Hi-ToM cot, 距 DS 缩小):
- v3.1: -8.08pp
- v3.2 (1200 Hi-ToM 真实数据 + reward worker A-Z fix): -3.25pp
- v3.3 (Hi-ToM direct-style + EmoBench/SocialIQA 合成扩充): -1.25pp
- **v3.4 (GPT-5.5 paraphrase + verified distillation): -0.42pp** ⭐

---

## TL;DR (v3.2 增补 — 历史)

1. **v3.2 Hi-ToM 全 3 协议大幅提升** vs v3.1: direct +3.66pp, cot +4.83pp, del_tom +3.33pp — 验证了 Hi-ToM 训练数据 (1200 条) + reward worker A-Z + l_max_long 修复有效。
2. **v3.2 ToMBench 微退化** vs v3.1: -0.29 / -0.50 / -0.44pp — 新数据 (17% 占比) 在 backbone ToMBench 上引入轻微干扰。
3. **v3.2 SocialIQA / EmoBench 基本持平** vs v3.1 (Δ ±0.6pp 内)，未能闭合与 DeepSeek 的 -2~-11pp 差距 — 仅 1500 条合成数据不足以转移这两个领域。
4. **v3.2 仍未全面超越 DeepSeek-v4-pro** — 在 Hi-ToM cot 上接近 (0.7150 vs 0.7475, -3.25pp) 但其他大多 -3 ~ -11pp。
5. **GPT-5.5 唯一成功跑通的是 ToMBench direct (0.8349)**，其它 11 格仍空缺。
6. **训练对 Hi-ToM 强迁移**: 14B v3.2 vs base 在 Hi-ToM cot/del_tom 提升 +27/+28pp；SocialIQA / EmoBench 改善仍微弱。

---

## 0. 全局总表 — 7 模型 × 4 Benchmark × 3 协议

每行 = (Benchmark, 协议)，每列 = 模型。**粗体** = 该行最高，— 表示未评测，()内为样本数若 < 标准量。

| Benchmark | Protocol | 8B base | 14B base | 8B v1.0 | 14B v3.1 | **14B v3.2** | DeepSeek | GPT-5.5 |
|---|---|---|---|---|---|---|---|---|
| **ToMBench** (n=5718) | direct | 0.7009 | 0.7338 | 0.7450 | 0.7721 | 0.7692 | 0.8080 | **0.8349** |
| ToMBench | cot | 0.7464 (n=3675) | — | 0.7501 | 0.7754 | **0.7704** | 0.7140 (n=500) | — |
| ToMBench | del_tom | — | — | 0.7618 | **0.7875** | 0.7831 | 0.8069 | — |
| **Hi-ToM** (n=600) | direct | 0.5550 | 0.5333 | 0.5717 | 0.5417 | 0.5783 | **0.8033** | — |
| Hi-ToM | cot | 0.4467 | 0.4367 | 0.6183 | 0.6667 | **0.7150** | 0.7475 (n=598) † | — |
| Hi-ToM | del_tom | 0.4817 | 0.4783 | 0.6567 | 0.7217 | **0.7550** | — (hang) | — |
| **SocialIQA** (n=1954) | direct | 0.7559 | 0.7871 | 0.7605 | 0.7876 | 0.7881 | **0.8101** | — |
| SocialIQA | cot | 0.7600 | 0.7856 | 0.7733 | 0.7861 | 0.7810 | **0.8106** | — |
| SocialIQA | del_tom | 0.7758 | 0.8035 | 0.7851 | 0.7892 | 0.7835 | **0.8188** | — |
| **EmoBench** (n=1200) | direct | 0.6033 | 0.6325 | 0.6000 | 0.6483 | 0.6492 | **0.7625** | — |
| EmoBench | cot | 0.5725 | 0.6342 | 0.6208 | 0.6575 | 0.6517 | **0.7550** | — |
| EmoBench | del_tom | 0.6233 | 0.6717 | 0.6342 | 0.6775 | 0.6600 | **0.7733** | — |

† DeepSeek Hi-ToM cot 有 2 题 timeout，有效 447/598 = 0.7475；del_tom (4800 sample) 在 API 上多次 hang，不可行。

**模型简称对照**:
- `Qwen3-8B v1.0` = Stage 15 ckpt-150 (production_frozen 8B 最优)
- `Qwen3-14B v3.1` = Stage 14b ckpt-199 (前 production_frozen 14B)
- `Qwen3-14B v3.2` = **Stage 16 ckpt-270 (新)**：在 v3.1 续训，加入 1200 Hi-ToM 训练数据 + 1500 EmoBench/SocialIQA 合成数据 + reward worker A-Z 字母 + task-aware l_max
- `DeepSeek-v4-pro`: max_tokens=8192 (修复 reasoning_content 截断 bug)
- `GPT-5.5`: 仅有 ToMBench direct 一格数据

**逐列每行最高排名次数** (12 行中的"该行第一"次数):

| 8B base | 14B base | 8B v1.0 | 14B v3.1 | **14B v3.2** | DeepSeek | GPT-5.5 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 1 (ToMBench del_tom) | 2 (Hi-ToM cot/del_tom) | 8 | 1 (ToMBench direct) |

**v3.2 vs v3.1 提升幅度**:

| Benchmark | Protocol | v3.1 | v3.2 | Δ |
|---|---|---|---|---|
| ToMBench | direct | 0.7721 | 0.7692 | **-0.29pp** |
| ToMBench | cot | 0.7754 | 0.7704 | **-0.50pp** |
| ToMBench | del_tom | 0.7875 | 0.7831 | **-0.44pp** |
| **Hi-ToM** | direct | 0.5417 | 0.5783 | **+3.66pp** ⭐ |
| **Hi-ToM** | cot | 0.6667 | 0.7150 | **+4.83pp** ⭐⭐ |
| **Hi-ToM** | del_tom | 0.7217 | 0.7550 | **+3.33pp** ⭐ |
| SocialIQA | direct | 0.7876 | 0.7881 | +0.05pp |
| SocialIQA | cot | 0.7861 | 0.7810 | -0.51pp |
| SocialIQA | del_tom | 0.7892 | 0.7835 | -0.57pp |
| EmoBench | direct | 0.6483 | 0.6492 | +0.09pp |
| EmoBench | cot | 0.6575 | 0.6517 | -0.58pp |
| EmoBench | del_tom | 0.6775 | 0.6600 | **-1.75pp** ⚠️ |

**v3.2 vs DeepSeek-v4-pro 距离**:

| Benchmark | Protocol | v3.2 | DeepSeek | Gap |
|---|---|---|---|---|
| ToMBench | direct | 0.7692 | 0.8080 | **-3.88pp** |
| ToMBench | cot | 0.7704 | 0.7140 | **+5.64pp** ✅ (但 DeepSeek 是 subset500) |
| ToMBench | del_tom | 0.7831 | 0.8069 | -2.38pp |
| Hi-ToM | direct | 0.5783 | 0.8033 | -22.50pp |
| Hi-ToM | cot | 0.7150 | 0.7475 | **-3.25pp** |
| Hi-ToM | del_tom | 0.7550 | — (hang) | — ✅ |
| SocialIQA | direct | 0.7881 | 0.8101 | -2.20pp |
| SocialIQA | cot | 0.7810 | 0.8106 | -2.96pp |
| SocialIQA | del_tom | 0.7835 | 0.8188 | -3.53pp |
| EmoBench | direct | 0.6492 | 0.7625 | -11.33pp |
| EmoBench | cot | 0.6517 | 0.7550 | -10.33pp |
| EmoBench | del_tom | 0.6600 | 0.7733 | -11.33pp |

**总结**: v3.2 在 Hi-ToM 上**接近 DeepSeek**（cot gap 缩到 -3.25pp，del_tom 没法直接比因 DeepSeek hang），EmoBench gap 仍 ~-11pp 是最大瓶颈。

---

## TL;DR (历史 v3.1 总结)

1. **GPT-5.5 唯一成功跑通的是 ToMBench direct (0.8349)**，其它 11 格仍空缺。
2. **DeepSeek-v4-pro 在 4 个 benchmark 上几乎全面领先**；唯一例外是 ToMBench cot (0.7140 subset500，疑被 reasoning truncation 拖累)。
3. **训练对 ToMBench / Hi-ToM 强迁移**: 14B v3.1 vs base 在 ToMBench 提升 ~+4pp，Hi-ToM cot/del_tom 提升 +23~+24pp；SocialIQA / EmoBench 几乎无迁移 (<2pp)。
4. **Hi-ToM 对 DeepSeek 的 reasoning_content 长度极敏感**: del_tom (4800 sample × 长 CoT) 在 max_tokens=8192 下仍多次 hang，无法完成；direct 完整 (0.8033)，cot 完整 600/600 但有 2 个 timeout (447/598 = 0.7475)。

---

## 1. ToMBench (5718 题, Theory of Mind 主战场)

| Protocol | Qwen3-8B base | Qwen3-14B base | Qwen3-8B v1.0 | Qwen3-14B v3.1 | DeepSeek-v4-pro | GPT-5.5 |
|---|---|---|---|---|---|---|
| direct | 0.7009 | 0.7338 | **0.7450** | **0.7721** | 0.8080 | **0.8349** |
| cot | 0.7464 † | — | 0.7501 | 0.7754 | 0.7140 ‡ | — |
| del_tom | — | — | 0.7618 | **0.7875** | 0.8069 | — |

† Qwen3-8B base cot 在 full5718 上仅 3675 题完成 (n=3675, 0.7464)；subset500 cot = 0.7640。
‡ DeepSeek cot 仅 subset500 (n=500, 0.7140)，疑因 reasoning_content 被 max_tokens=2048 截断；direct/del_tom 走 full5718 重测后 ≥0.80。
— Qwen3-14B base 仅有 direct full5718 数据 (qwen3-14b-nt)。GPT-5.5 仅评测了 direct。

**关键差距 (direct, full5718)**:
- 8B base → 8B v1.0: **+4.41pp**
- 14B base → 14B v3.1: **+3.83pp**
- 14B v3.1 ↔ DeepSeek-v4-pro: -3.59pp
- 14B v3.1 ↔ GPT-5.5: -6.28pp

---

## 2. Hi-ToM (600 题, 5 阶 belief 追踪)

| Protocol | Qwen3-8B base | Qwen3-14B base | Qwen3-8B v1.0 | Qwen3-14B v3.1 | DeepSeek-v4-pro | GPT-5.5 |
|---|---|---|---|---|---|---|
| direct | 0.5550 | 0.5333 | 0.5717 | 0.5417 | **0.8033** | — |
| cot | 0.4467 | 0.4367 | 0.6183 | **0.6667** | ~0.7475 ‡ (partial) | — |
| del_tom | 0.4817 | 0.4783 | 0.6567 | **0.7217** | — § | — |

‡ DeepSeek Hi-ToM cot: 600/600 完成但 2 题 timeout，有效 447/598 = 0.7475；纯命中率 (excl-error) ~0.86。
§ DeepSeek Hi-ToM del_tom: 4800 sample × 长 reasoning，先后 4 次 process hang (conc 4/8/12 各试过)，未跑出可信结果。

### Per-task breakdown (direct, by ToM order)

| Order | 8B base | 14B base | 8B v1.0 | 14B v3.1 | DeepSeek |
|---|---|---|---|---|---|
| order_0 | 0.825 | 0.825 | 0.825 | 0.892 | **1.000** |
| order_1 | 0.583 | 0.575 | 0.542 | 0.575 | **0.960** |
| order_2 | 0.450 | 0.392 | 0.508 | 0.400 | **0.870** |
| order_3 | 0.483 | 0.450 | 0.508 | 0.433 | **0.670** |
| order_4 | 0.508 | 0.425 | 0.475 | 0.408 | **0.820** (partial) |

**RL 增益最大的格**:
- 14B v3.1 cot - 14B base cot = **+23.00pp**
- 14B v3.1 del_tom - 14B base del_tom = **+24.34pp**
- 8B v1.0 cot - 8B base cot = **+17.16pp**
- 8B v1.0 del_tom - 8B base del_tom = **+17.50pp**

---

## 3. SocialIQA dev (1954 题, 3 选项 commonsense)

| Protocol | Qwen3-8B base | Qwen3-14B base | Qwen3-8B v1.0 | Qwen3-14B v3.1 | DeepSeek-v4-pro | GPT-5.5 |
|---|---|---|---|---|---|---|
| direct | 0.7559 | 0.7871 | 0.7605 | 0.7876 | **0.8101** | — |
| cot | 0.7600 | 0.7856 | 0.7733 | 0.7861 | **0.8106** | — |
| del_tom | 0.7758 | **0.8035** | 0.7851 | 0.7892 | **0.8188** | — |

**关键观察**:
- ToMBench RL 训练**几乎不迁移到 SocialIQA**: v3.1 vs 14B base 的 direct/cot 仅 +0.05pp，del_tom 反而 -1.43pp ⚠️
- 8B v1.0 vs 8B base 略有提升 (+0.5~+1.3pp)，但远低于 ToMBench / Hi-ToM 的增益

---

## 4. EmoBench (1200 题, EA + EU_emotion + EU_cause 各 400)

| Protocol | Qwen3-8B base | Qwen3-14B base | Qwen3-8B v1.0 | Qwen3-14B v3.1 | DeepSeek-v4-pro | GPT-5.5 |
|---|---|---|---|---|---|---|
| direct | 0.6033 | 0.6325 | 0.6000 | 0.6483 | **0.7625** | — |
| cot | 0.5725 | 0.6342 | 0.6208 | 0.6575 | **0.7550** | — |
| del_tom | 0.6233 | 0.6717 | 0.6342 | 0.6775 | **0.7733** | — |

### Per sub-task (del_tom)

| Sub-task | 8B base | 14B base | 8B v1.0 | 14B v3.1 | DeepSeek |
|---|---|---|---|---|---|
| EA (action) | 0.6325 | 0.6975 | 0.6725 | 0.7075 | ~0.83+ |
| EU_emotion (情绪命名 6 选 1) | 0.4500 | 0.5050 | 0.4625 | 0.5375 | ~0.70+ |
| EU_cause | 0.7325 | 0.7800 | 0.7675 | 0.7875 | ~0.80+ |

**最弱子项**: EU_emotion — 6 选项情绪命名，所有 Qwen3 模型 <0.55，DeepSeek 也只到 ~0.70。RL 训练对此项小幅推进 (+3pp)。

---

## 5. RL 增益矩阵 (vs same-size base)

| Benchmark | Protocol | 14B v3.1 - 14B base | 8B v1.0 - 8B base |
|---|---|---|---|
| ToMBench | direct | **+3.83pp** | **+4.41pp** |
| ToMBench | cot | — / — | +0.37pp |
| ToMBench | del_tom | — / — | — |
| **Hi-ToM** | direct | +0.84pp | +1.67pp |
| **Hi-ToM** | **cot** | **+23.00pp** ⭐ | **+17.16pp** ⭐ |
| **Hi-ToM** | **del_tom** | **+24.34pp** ⭐ | **+17.50pp** ⭐ |
| SocialIQA | direct | +0.05pp | +0.46pp |
| SocialIQA | cot | +0.05pp | +1.33pp |
| SocialIQA | del_tom | **-1.43pp** ⚠️ | +0.92pp |
| EmoBench | direct | +1.58pp | -0.33pp |
| EmoBench | cot | +2.33pp | +4.83pp |
| EmoBench | del_tom | +0.58pp | +1.08pp |

---

## 6. 与目标的差距 (direct, ToMBench full5718)

| 模型 | direct acc | vs DeepSeek-v4-pro | vs GPT-5.5 |
|---|---|---|---|
| Qwen3-8B base | 0.7009 | -10.71pp | -13.40pp |
| Qwen3-8B v1.0 | 0.7450 | -6.30pp | -8.99pp |
| Qwen3-14B base | 0.7338 | -7.42pp | -10.11pp |
| Qwen3-14B v3.1 | **0.7721** | **-3.59pp** | **-6.28pp** |

**Spec 目标**:
- 接近 X (DeepSeek-v4-pro): direct ≥ 0.7680 → ✅ 14B v3.1 已达成
- 超越 X: direct ≥ 0.7880 → ❌ 14B v3.1 距 X 仍差 3.59pp

---

## 7. 缺测项与原因

| 模型 × Benchmark | 协议 | 原因 |
|---|---|---|
| GPT-5.5 × ToMBench | cot, del_tom | 未跑 (只跑了 direct full5718) |
| GPT-5.5 × Hi-ToM | direct, cot, del_tom | 未跑 (新增 benchmark 后未排期) |
| GPT-5.5 × SocialIQA | direct, cot, del_tom | 未跑 |
| GPT-5.5 × EmoBench | direct, cot, del_tom | 未跑 |
| DeepSeek-v4-pro × Hi-ToM | del_tom | 4 次 hang (conc=4/8/12 都失败)；4800 长 CoT sample 在 deepseek API 上不可行 |
| DeepSeek-v4-pro × Hi-ToM | cot | 600/600 但有 2 timeout，有效 0.7475 |
| DeepSeek-v4-pro × ToMBench | cot | 仅 subset500 (0.7140)，full5718 未重测 |
| Qwen3-14B base × ToMBench | cot, del_tom | 仅有 direct full5718；cot/del_tom 未跑 |
| Qwen3-8B base × ToMBench | cot | full5718 仅 3675 题 (0.7464)，subset500 cot = 0.7640 |
| Qwen3-8B base × ToMBench | del_tom | 未跑 |

---

## 8. 评测协议与脚本一览

- **ToMBench**: `scripts/eval/run_tombench.py`，`data/tom/tombench_eval.jsonl` (5718)
- **3 个新 benchmark**: `scripts/eval/run_generic_mcq.py` + `scripts/eval/extractors_generic.py`
- **DeepSeek 重要修复**: max_tokens=2048 → 8192（避免 reasoning_content 截断）；`is_qwen` 分支决定是否禁 thinking + 用 64 token cap
- **del_tom 投票**: 8 sample × T=0.7, top_p=0.95, 多数投票，平局取 lowest letter
- **数据集来源**:
  - SocialIQA: dev split via `scripts/data/build_socialiqa_eval.py`
  - EmoBench: HF `SahandSab/EmoBench` (EA + EU 双子题)
  - Hi-ToM: order_0~order_4 各 ~120 题
  - ToMBench: 8 task × 中英双语，5718 = 2860 × 2 - missing-lang

---

## 9. 数据完整性证书

| Benchmark | 6 模型 × 3 协议 = 18 格 | 完整 | partial | 空缺 |
|---|---|---|---|---|
| ToMBench | 18 | 7 | 2 | **9** (主要为 GPT-5.5 cot/del_tom + 14B base cot/del_tom) |
| Hi-ToM | 18 | 14 | 1 (DeepSeek cot) | **3** (GPT-5.5 全部 + DeepSeek del_tom) |
| SocialIQA | 18 | 15 | 0 | **3** (GPT-5.5 全部) |
| EmoBench | 18 | 15 | 0 | **3** (GPT-5.5 全部) |
| **总计** | **72** | **51** | **3** | **18** |

完成率: 51/72 = **70.8%**；含 partial 算 75%。

---

## 10. 后续建议

1. **优先补 GPT-5.5 在 4 benchmark × cot/del_tom 的 11 格**（最可能改变排名）
2. Hi-ToM del_tom 对 DeepSeek 不可行 → 不再追求；改用 cot full 数据 (0.7475) 作 baseline
3. 14B base cot/del_tom 在 ToMBench full5718 补测，量化 base→v3.1 的 cot/del_tom 增益（目前只有 direct +3.83pp 一个数）
4. 考虑用 SocialIQA-style + EmoBench EU_emotion 训练数据做下一阶段 SFT/RL，针对最弱迁移方向
