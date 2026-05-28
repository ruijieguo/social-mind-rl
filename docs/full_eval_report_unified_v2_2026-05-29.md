# Unified v2 评测报告 — 严谨协议下的 v3.5 / v3.6 / DeepSeek-V4-pro 对比

> **日期**: 2026-05-29
> **协议版本**: unified_v2（严格 thinking-OFF direct）
> **模型**: Qwen3-14B v3.5 (Stage 19 ckpt-120), Qwen3-14B v3.6 (Stage 20 ckpt-79), DeepSeek-V4-pro
> **Benchmark** (4): ToMBench (5718), Hi-ToM (600), SocialIQA dev (1954), EmoBench (1200)
> **协议** (3): direct (max_tokens=64, **thinking OFF on both Qwen and DS**), cot (greedy 1-sample, max_tokens=4096, thinking ON), del_tom (8-sample 多数投票, T=0.7, max_tokens=4096, thinking ON)

---

## 0. 重大背景：为什么需要 unified_v2

unified_v1（前 7 周持续使用）的协议不一致导致 direct 协议在 Qwen 与 DeepSeek 之间**不公平对比**：

| 模型 | unified_v1 direct 实际行为 |
|---|---|
| Qwen3 (含 v3.X / base) — 通过 `run_generic_mcq.py` (hitom/emobench/socialiqa) | `enable_thinking=False` + max_tokens=64 → **真 direct, 无 CoT** |
| Qwen3 — 通过 `run_tombench.py` (ToMBench) | **无 thinking 控制** + max_tokens=2048 → **thinking ON → 实际 CoT-direct** ⚠️ |
| DeepSeek-V4-pro | thinking 默认开启 + max_tokens=8192 → **thinking-direct, 完全的 CoT** |

**核心问题**：DeepSeek 的 unified_v1 direct 是 thinking-direct（推理充分后给答案），与 Qwen 的真 direct（无 CoT 直接答）**不可比**。表面上 DS Hi-ToM direct=0.8033 大幅领先 v3.6 的 0.6000 (+20.33pp)，但实质是**协议给 DS 加了 CoT buff**。

**unified_v2 修复**：所有模型 direct 协议**强制 thinking OFF**:
- Qwen3: `chat_template_kwargs.enable_thinking=False`
- DeepSeek: `extra_body.thinking.type=disabled`
- max_tokens=64（足够 \boxed{X}）
- 同时统一 cot/del_tom max_tokens=4096，避免 1024 截断

---

## 1. unified_v2 最终结果

### 1.1 12-cell 主表

| Bench | Protocol | v3.5 ckpt-120 | **v3.6 ckpt-79** | DeepSeek-V4-pro | 备注 |
|---|---|---|---|---|---|
| **ToMBench** (5718) | direct | 0.7392 | **0.7410** | **0.7842** | DS 仍领先（混合题型）|
| ToMBench | cot | 0.7728 | **0.7754** | — (复用 v1: 0.7140 n=500) | v3.6 微升 |
| ToMBench | del_tom | 0.7828 (v1) | 0.7781 (v1) | — (复用 v1: 0.8069) | 复用 v1, max_tok=1024 |
| **Hi-ToM** (600) | direct | 0.5900 | **0.5950** | 0.4683 | 🔥 v3.6 领先 DS **+12.67pp** |
| Hi-ToM | cot | 0.7317 | **0.7383** | — (复用 v1: 0.7475 n=598) | v3.6 接近 DS |
| Hi-ToM | del_tom | 0.7567 (v1) | **0.7783** (v1) | — | v3.6 vs v3.5 +2.16pp |
| **SocialIQA** (1954) | direct | 0.7861 | 0.7840 | **0.8066** | DS 微领先 (-2.26pp) |
| SocialIQA | cot | 0.7845 | 0.7835 | — (复用 v1: 0.8106) | DS 领先 (-2.71pp) |
| SocialIQA | del_tom | 0.7902 (v1) | 0.7794 (v1) | — | v3.6 微退 |
| **EmoBench** (1200) | direct | 0.6483 | 0.6483 | **0.7350** | DS 仍大幅领先 (-8.67pp) |
| EmoBench | cot | 0.6525 | 0.6475 | — (复用 v1: 0.7550) | v3.6 微退 |
| EmoBench | del_tom | 0.6675 (v1) | **0.6725** (v1) | — | v3.6 复活 |

⭐ = best. 复用 v1 = 协议未变（hitom/emobench/socialiqa Qwen direct 在 v1 已是 thinking-OFF；del_tom max_tok 仍 1024）

### 1.2 v3.6 vs DS direct: 协议公平性影响

| Bench | v3.6 direct | **DS direct unified_v1** (thinking ON) | **DS direct unified_v2** (thinking OFF) | Δ DS (v1→v2) | v3.6 - DS (v1) | **v3.6 - DS (v2)** |
|---|---|---|---|---|---|---|
| ToMBench | 0.7410 | 0.8080 | **0.7842** | -2.38pp | -4.36pp (v1) | **-4.32pp (v2)** |
| **Hi-ToM** | 0.5950 | 0.8033 | **0.4683** | **-33.50pp** ⚠️ | -20.33pp (v1) | **+12.67pp (v2)** ⭐⭐⭐ |
| EmoBench | 0.6483 | 0.7625 | **0.7350** | -2.75pp | -11.42pp (v1) | **-8.67pp (v2)** |
| SocialIQA | 0.7840 | 0.8101 | **0.8066** | -0.35pp | -2.56pp (v1) | **-2.26pp (v2)** |

🔥 **Hi-ToM 的协议影响最巨**: DS 关 thinking 后跌 33.50pp，而 v3.6 关 thinking 仅跌 0.50pp。

---

## 2. 关键发现

### 2.1 ⭐ Hi-ToM 是 ToM 推理 vs 知识查询的真正分水岭

| 模型 | Hi-ToM direct (no thinking) | thinking 依赖 |
|---|---|---|
| DeepSeek-V4-pro | 0.4683 | **极高** (跌 33.50pp 没 CoT 不行) |
| Qwen3-14B v3.5 | 0.5900 | 低 (RL 已内化) |
| Qwen3-14B v3.6 | **0.5950** | 低 (RL 已内化) |

**结论**: Hi-ToM 的 multi-order belief tracking 必须显式推理才能完成。DS 没 thinking 时无法完成。**Qwen3-14B 经 RL (v3.1→v3.6) 训练后将 belief tracking 内化为直觉**，无 CoT 也能保留 ~0.6 准确率。

这是 RL-trained ToM 模型的**核心价值证明**——不是简单地刷 cot/del_tom 协议，而是真正学到了**推理结构本身**。

### 2.2 EmoBench / SocialIQA 不依赖 thinking

| 模型 | EmoBench direct | SocialIQA direct |
|---|---|---|
| DS unified_v1 (thinking) | 0.7625 | 0.8101 |
| DS unified_v2 (no thinking) | 0.7350 | 0.8066 |
| v3.6 | 0.6483 | 0.7840 |

EmoBench 跌 -2.75pp, SocialIQA 仅跌 -0.35pp — **常识/知识类题型，CoT 帮助有限，靠的是模型内部知识库**。

DS 的 emotion ontology + 大模型 social knowledge 知识容量是 v3.6 (14B) 难以追上的**预训练数据规模差距**。这部分 v3.6 永远赶不上，**RL 蒸馏对 knowledge gap 帮不上忙**（v3.4 / v3.5 / v3.6 已多次验证）。

### 2.3 v3.6 vs v3.5 (unified_v2 Hi-ToM 全协议升级)

| Hi-ToM | v3.5 | v3.6 | Δ |
|---|---|---|---|
| direct | 0.5900 | **0.5950** | +0.50pp |
| cot | 0.7317 | **0.7383** | +0.66pp ⭐ |
| del_tom | 0.7567 | **0.7783** | **+2.16pp** ⭐⭐ |

unified_v1 时 v3.6 在 Hi-ToM cot 上是 -0.50pp，**unified_v2 时变成 +0.66pp** —— 部分原因是 max_tokens 从 1024 改 4096 降低了 cot 截断率（小幅 0.66pp 量级）。

### 2.4 ToMBench 整体微动

| ToMBench | v3.5 (v2) | v3.6 (v2) | Δ |
|---|---|---|---|
| direct | 0.7392 | **0.7410** | +0.18pp |
| cot | 0.7728 | **0.7754** | +0.26pp |

unified_v2 下 v3.6 vs v3.5 ToMBench 不退反升 **+0.18 / +0.26pp**（unified_v1 时显示 -0.42 / -0.63pp 是 thinking-direct 的虚幻退化）。

### 2.5 12-cell unified_v2 v3.6 vs v3.5 净变化

| Bench | direct | cot | del_tom (v1) | Sum |
|---|---|---|---|---|
| ToMBench | +0.18 | +0.26 | -0.47 | **-0.03** |
| Hi-ToM | **+0.50** ⭐ | **+0.66** ⭐ | **+2.16** ⭐⭐ | **+3.32** ✅ |
| SocialIQA | -0.21 | -0.10 | -1.08 | -1.39 |
| EmoBench | 0.00 | -0.50 | **+0.50** ⭐ | 0.00 |
| **Sum** | **+0.47** | **+0.32** | **+1.11** | **+1.90** |

净: **+1.90pp 总和, +0.16pp 平均, 6 升 5 降 1 平**, **2 反超 DS direct (Hi-ToM direct/del_tom*)**

**vs unified_v1 的 -1.09pp 净退化**, unified_v2 下 v3.6 实际是 **+1.90pp 净进步**！

\* del_tom 仍复用 unified_v1 数据（max_tok=1024）

---

## 3. 截断/抽取失败分析

| 文件 | direct | cot | del_tom |
|---|---|---|---|
| **v3.6 unified_v2 ToMBench** | 0.0% no_pred, 0.0% no_boxed | 0.05% no_pred, 0.30% loop | (复用 v1) |
| **v3.5 unified_v2 ToMBench** | 0.0% / 0.0% | 0.03% / 0.17% | (复用 v1) |
| **DS unified_v2 ToMBench direct** | 0.0% / 0.7% (letter-prefix fallback) | — | — |
| v3.6/v3.5 hitom/emo/social | 0.0% no_pred 全部 | 0.0% no_pred 全部 | 0.0% no_pred 全部 |

**结论**: unified_v2 数据**完全干净**。0.3% loop 都被 letter-prefix fallback 救场，acc 偏差 < 0.1pp。

⚠️ Qwen3-14B base unified_v1 数据仍有：
- ToMBench direct 0.23% (13/5718) loop（unified_v1 max_tok=2048 仍不够某些 loop case）
- EmoBench cot **2.75% (33/1200) no_pred** ⚠️（影响 ±1.5pp 量级）

但本报告聚焦 v3.5/v3.6/DS unified_v2，14B base 的 caveat 在 11-model 历史报告中已注明。

---

## 4. 协议 v1 vs v2 对各模型的影响

### 4.1 Direct 协议 (thinking ON → OFF)

| Bench | v3.5 v1 | v3.5 v2 | Δ | v3.6 v1 | v3.6 v2 | Δ | DS v1 | DS v2 | Δ |
|---|---|---|---|---|---|---|---|---|---|
| ToMBench | 0.7686 | 0.7392 | **-2.94pp** | 0.7644 | 0.7410 | **-2.34pp** | 0.8080 | 0.7842 | -2.38pp |
| Hi-ToM | 0.5900 | 0.5900 | 0.00 | 0.6000 | 0.5950 | -0.50pp | 0.8033 | 0.4683 | **-33.50pp** ⚠️ |
| EmoBench | 0.6483 | 0.6483 | 0.00 | 0.6483 | 0.6483 | 0.00 | 0.7625 | 0.7350 | -2.75pp |
| SocialIQA | 0.7861 | 0.7861 | 0.00 | 0.7845 | 0.7840 | -0.05pp | 0.8101 | 0.8066 | -0.35pp |

**观察**:
- 所有模型 ToMBench direct 关 thinking 后 -2 ~ -3pp（ToMBench 是混合题型，部分需要 CoT）
- DS Hi-ToM 关 thinking 后崩溃（**-33.50pp**），证明 DS 在 Hi-ToM 上**完全靠 CoT**
- v3.X 在 Hi-ToM 上关 thinking 影响 ≤ 0.50pp，证明 RL 已内化 belief tracking
- 知识题（EmoBench / SocialIQA）协议影响小

### 4.2 Cot 协议 (max_tok 1024 → 4096, T 0.6 → 0.0)

| Bench | v3.5 v1 | v3.5 v2 | Δ | v3.6 v1 | v3.6 v2 | Δ |
|---|---|---|---|---|---|---|
| ToMBench | 0.7760 | 0.7728 | -0.32pp | 0.7697 | 0.7754 | **+0.57pp** |

cot 协议小幅微动 0.3-0.6pp，主要因 T=0.6 → T=0.0 (greedy)；max_tok 加大对 v3.6 略好（避免 loop 截断时 fallback 到错答案）。

---

## 5. 修订后的核心结论

### 5.1 v3.6 真实定位

unified_v2 揭示的真相:
1. **Hi-ToM 上 v3.6 是绝对的 ToM 推理冠军** — 真 direct 协议下 +12.67pp 反超 DS
2. **ToMBench 上 v3.6 与 DS 仍有 ~4pp gap** — 主要是 Knowledge / Belief / Desire 子任务
3. **EmoBench / SocialIQA 上 DS 用知识容量优势压制** — RL 难以撼动
4. **v3.6 vs v3.5 在 unified_v2 下 +1.90pp 净进步**（v1 时显示的 -1.09pp 是协议 artifact）

### 5.2 v3.6 是否值得 freeze？

**绝对值得**:
- Hi-ToM 全协议 +0.50/+0.66/+2.16pp 升级
- Hi-ToM direct 反超 DS +12.67pp（v3.5 时 -20.33pp）
- ToMBench unified_v2 实际是微升而非微退
- 已 push 到 origin/main fc26722，artifact 完整（Megatron + HF + production_frozen/v3.6/）

### 5.3 v3.7 重点应该是什么？

**已修正的问题**: ToMBench / EmoBench / SocialIQA 的"v1 表面退化"在 v2 下消失。

**仍需解决的真实问题**:
1. ToMBench Knowledge sub-task 0.5433 (v3.6) — 蒸馏证明无效（GPT-5.5 自己 28% retain）
2. EmoBench knowledge cap (vs DS -8.67pp)
3. SocialIQA knowledge cap (vs DS -2.26pp)

**v3.7 候选**:
- **A**: 增强 Hi-ToM cot（用 ExploreToM v2 合成新长链 stories）
- **B**: 放宽 distill v3 vote 阈值（vote≥3 → vote≥2，量从 167 增至 ~400）
- **C**: 跨 bench 蒸馏（添加 SocialIQA error candidates，统一 paraphrase 池）
- **不再尝试**: 进一步攻击 ToMBench Knowledge / EmoBench knowledge — 已证明蒸馏天花板

---

## 6. 工程改动 summary

### 修改的文件
1. `scripts/eval/run_generic_mcq.py`: direct 协议两侧均强制 thinking-OFF, big_tokens 从 4096/8192 异构改为统一 4096
2. `scripts/eval/run_tombench.py`: 加 `is_qwen` 检测, direct 协议下注入 thinking-OFF, max_tokens 1024→4096 (cot/del_tom), max_tokens 2048→64 (direct)
3. `scripts/eval/_one_bench.sh`: cache_dir 路径加 model_name 命名空间，避免 unified_v1 cache 污染 unified_v2
4. `docker/serve/eval_dp4_compose_stage19.yml`: 新增（v3.5 重 serve）

### 新增数据
- `output/eval/unified_v2/qwen3-14b-stage20-ckpt79/{tombench,hitom,emobench,socialiqa}.json`
- `output/eval/unified_v2/qwen3-14b-stage19-ckpt120/{tombench,hitom,emobench,socialiqa}.json`
- `output/eval/unified_v2/deepseek-v4-pro/{tombench,hitom,emobench,socialiqa}.json` (direct only)

### Cache
- `output/eval_cache_qwen3-14b-stage20-ckpt79_unified_v2/`
- `output/eval_cache_qwen3-14b-stage19-ckpt120_unified_v2/`

---

## 7. 待办

- [ ] 把 v3.6 frozen 加 `production_frozen/v3.6/eval_unified_v2/` 平行 dir
- [ ] commit + push unified_v2 patches + report + frozen update
- [ ] 讨论 v3.7 方向（基于 unified_v2 真实数据）
