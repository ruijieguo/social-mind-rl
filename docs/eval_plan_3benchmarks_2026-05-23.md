# 评测计划：v3.1 / 8b-v1.0 / deepseek-v4-pro × {SocialIQA, EmoBench, Hi-ToM} × {direct, cot, del_tom}

> **目标**: 在 3 个新 benchmark 上扩展评测，验证 v3.1 / 8b-v1.0 / deepseek-v4-pro 在 ToMBench 之外的泛化能力。
> **创建**: 2026-05-23, 待用户确认后执行。

## 1. Benchmark 概况

### 1.1 SocialIQA
- **来源**: `allenai/social_i_qa` (HuggingFace), AAAI 2019
- **格式**: 3-option MCQ (answerA/B/C, label "1"/"2"/"3")
- **Schema**: `context, question, answerA, answerB, answerC, label`
- **数据量**: 训练集 33k+，dev 1954
- **现有支持**: `scripts/data/build_socialiqa.py` 已存在 (55 行)，需调整：
  - 已有逻辑把 3-opt 转成 4-opt (加 "None of the above")，但策略可疑，可能干扰评测
  - **决策**: 改为评测 dev set 1954 题，保留原始 3-option，扩展 run_tombench 支持 3-opt
- **采样**: dev 全 1954 题（1 次完整评测）

### 1.2 EmoBench
- **来源**: `SahandSab/EmoBench` (HuggingFace), ACL 2024
- **格式**: 4-option MCQ, label = 选项原文（不是字母）
- **Schema**: `qid, language, category, question type, scenario, subject, choices, label`
- **数据量**: 800 (400 EU + 400 EA, 双语)
- **决策**: 评测全部 800 题
- **特别处理**: label 是文本 → 转为 "A/B/C/D" 字母 gold

### 1.3 Hi-ToM
- **来源**: ToM-RL author's `Hi_ToM_cleaned.csv` (本地已有, 600 题), EMNLP 2023
- **格式**: 18+ option MCQ (A-R 字母, location 选项)
- **Schema**: `story, question, choices, answer, question_order` (0-4 阶 ToM)
- **数据量**: 600 题（5 阶 × 120 题/阶）
- **关键决策**:
  - **Option A**: 保留原始 18-opt（ToMBench eval framework 是 4-opt only，需大改）
  - **Option B (推荐)**: **dynamic prompting** — 把 18 个选项原文插入 prompt，让模型生成原文 answer，extractor 字符串匹配 gold answer。这种方式不需要 A-D 字母约束。
  - **Option C**: 把 18-opt 缩减到 4-opt（gold + 3 distractor），但破坏原始 benchmark
- **采样**: 全 600 题

## 2. 评测矩阵

| 模型 | host | benchmark | 协议 | 题量 |
|---|---|---|---|---|
| 14B v3.1 ckpt-199 | TRAIN | SocialIQA | direct,cot,del_tom | 1954 × 3 |
| 14B v3.1 ckpt-199 | TRAIN | EmoBench | direct,cot,del_tom | 800 × 3 |
| 14B v3.1 ckpt-199 | TRAIN | Hi-ToM | direct,cot,del_tom | 600 × 3 |
| 8B v1.0 ckpt-150 | TRAIN | (同上 3 benchmarks × 3 protocols) | | 同上 |
| deepseek-v4-pro | dev | SocialIQA | del_tom only | 1954 × 8 = 15.6k API calls |
| deepseek-v4-pro | dev | EmoBench | del_tom only | 800 × 8 = 6.4k |
| deepseek-v4-pro | dev | Hi-ToM | del_tom only | 600 × 8 = 4.8k |

**del_tom only for deepseek**: del_tom 是最强的协议，cost 最高。direct/cot 在 ToMBench 上 deepseek 已知数据。如果用户要 direct/cot，加 ~20k API calls。

**总评测次数 (本地 vLLM)**: 2 模型 × 3 benchmarks × 3 协议 = 18 次评测

## 3. 资源 & ETA

### 3.1 GPU 分配 (TRAIN, 8× H800)

| GPU | 用途 |
|---|---|
| 0-1 | 14B v3.1 vLLM (TP=2, ~40 GB) |
| 2-3 | 8B v1.0 vLLM (TP=1, ~16 GB) — 可只用 GPU 2 |
| 4-7 | 备用，或并行第二份 vLLM 加速 |

**优化**: 用 TP=1 在 GPU 0/1/2/3 各跑一份模型实例，4 个 client 并发 → 但只有 2 个目标模型，可能 GPU 重复加载。简单方案 = 1× 14B + 1× 8B。

### 3.2 预估时间

每个 protocol per benchmark ETA:
- **direct** (1 sample/q): 5-10 min
- **cot** (1 sample/q): 10-15 min
- **del_tom** (8 samples/q): 30-60 min

**Per model 总计**:
- SocialIQA (1954 q): direct 8min + cot 12min + del_tom 50min = ~70 min
- EmoBench (800 q): direct 4min + cot 6min + del_tom 25min = ~35 min
- Hi-ToM (600 q): direct 3min + cot 5min + del_tom 20min = ~30 min
- **每个模型: ~135 min**

**TRAIN total** (2 模型并行): ~135 min ≈ **2.5h**

**dev (deepseek del_tom only)**: 4.8k + 6.4k + 15.6k = 26.8k API calls @ 32 concurrency, ~1.2/s effective → ~6h. **Run in parallel with TRAIN evals.**

## 4. 工作分解

### 4.1 数据准备 (~1h)

需要 4 个新脚本/适配：

1. **`scripts/eval/build_socialiqa_eval.py`**: 从 HF 拉 dev split → 转成统一 schema
   - 输入: `allenai/social_i_qa`
   - 输出: `data/eval/socialiqa_eval.jsonl` (1954 records)
   - Schema: `question_id, source=socialiqa, language=en, task, story=context, question, opt_a/b/c, gold (A/B/C)`

2. **`scripts/eval/build_emobench_eval.py`**: 从 HF 拉 EmoBench → 转成统一 schema
   - 输入: `SahandSab/EmoBench`, `emotional_application` + `emotional_understanding`
   - 输出: `data/eval/emobench_eval.jsonl` (800 records)
   - Schema: `question_id, source=emobench, language=en/zh, task=EU/EA, story=scenario, question (隐含或合成), opt_a-d, gold`
   - **特别处理**: label 字符串 → 字母 gold

3. **`scripts/eval/build_hitom_eval.py`**: 从本地 `Hi_ToM_cleaned.csv` → 统一 schema
   - 输入: `data/tom/raw/hi_tom_gen/ToM-RL/data/cleaned_tom/raw/Hi_ToM_cleaned.csv`
   - 输出: `data/eval/hitom_eval.jsonl` (600 records)
   - Schema: `question_id, source=hitom, language=en, task=order_X, story, question, opt_a-r, gold (A-R)` 或 dynamic
   - **决策**: 保留 18-opt，extend run_tombench 支持任意数量选项

4. **`scripts/eval/run_tombench.py` 扩展**:
   - 支持任意数量选项 (3, 4, 18)
   - prompt template 动态填充选项
   - extractor 适配 (现在是固定 A/B/C/D，需扩展到 A-R)

### 4.2 评测执行 (~3h on TRAIN)

4 个 vLLM (2 模型 × 2 GPU 各) → 6 个 eval client (2 模型 × 3 benchmarks，每个 client 跑 3 protocols) 串行 protocol。

或简化：2 vLLM (1 个模型一个 vLLM) → 6 个 client 完全并行（每个 client 跑独立 benchmark + 串行 protocol）→ 但同一个 vLLM 被 3 个 client 同时调用要小心 throughput。

**推荐**:
- TRAIN: 2 个 vLLM (14B GPU 0-1, 8B GPU 2)
- 6 个 eval client 并行
- 每个 client: `--protocols direct,cot,del_tom`

### 4.3 deepseek dev 评测 (~6h, 并行)

3 个 dev container 各跑一个 benchmark（独立 API key 共享，但 3 个 process 把 concurrency 32 分摊）。或者 1 个 process 串行 3 benchmarks。

**推荐**: 1 个 process 串行（避免 rate limit 问题）。

## 5. 输出

每次评测生成:
- `output/eval/{model}_{benchmark}_full.json` (per-question results)
- `output/eval/{model}_{benchmark}_full.md` (aggregated table)

最终 cross-benchmark 比较表 → `output/eval/cross_benchmark_summary_2026-05-23.md`

## 6. 风险 & 决策点

| 风险 | 处理 |
|---|---|
| Hi-ToM 18-opt 让 prompt 长 | response_length 增到 512+，extractor 支持 A-R |
| EmoBench label 是文本不是字母 | build script 把文本 label → 字母 gold |
| SocialIQA 是 3-opt 与现有 4-opt 框架不兼容 | extend run_tombench 支持 3-opt |
| del_tom 在 18-opt 上的 voting 效果 | 取多数票选项，而不是 A/B/C/D |
| deepseek 6h 太长 | 默认只跑 del_tom；如需要 direct/cot 增加 |
| HF 网络问题 | dev 容器有 ModelScope 镜像，回退方案 |

## 7. 计划执行顺序

1. **数据准备**: 3 个 build script + run_tombench 扩展 (~1h)
2. **本地 smoke test**: 每个 benchmark 跑 50 题 with 8B v1.0 验证格式 (~10 min)
3. **TRAIN 启动**: 2 vLLM + 6 client 并行 (~2.5h)
4. **dev 启动**: deepseek del_tom × 3 benchmarks (~6h, 并行 TRAIN)
5. **聚合报告**: cross-benchmark summary

## 8. 用户确认点

请确认以下决策：

- [ ] **Hi-ToM 选项处理**: 推荐保留 18-opt 原始，dynamic prompting + 字母 gold (A-R) — 同意吗？
- [ ] **deepseek 协议**: 推荐 only del_tom（节省 ~3-4h cost）— 还是要全 3 协议？
- [ ] **SocialIQA 题量**: dev split 全 1954 题 — 同意吗？(也可缩到 500 加速)
- [ ] **是否做本地 smoke test**: 推荐做（10 min, 防 schema bug）— 同意吗？

确认后立即执行。
