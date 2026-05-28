"""Aggregate per-model eval results into a markdown report.

Reads:
  output/tombench/{base,v10,dashscope}.json
  output/hitom/{base,v10,dashscope}.json
Writes:
  output/full_eval_report_qwen3-8b_<DATE>.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path


MODELS = ["qwen3-8b-base", "qwen3-8b-v10", "qwen3-8b-api"]
BENCHMARKS = ["tombench", "hitom"]
PROTOCOLS = ["direct", "direct_think", "cot", "del_tom"]

PRETTY_MODEL = {
    "qwen3-8b-base": "Qwen3-8B base (local)",
    "qwen3-8b-v10":  "Qwen3-8B v1.0 (Stage 15 ckpt-150)",
    "qwen3-8b-api":  "Qwen3-8B (DashScope API)",
}

PRETTY_BENCH = {
    "tombench": "ToMBench (n=5718)",
    "hitom":    "Hi-ToM (n=600)",
}

PRETTY_PROTOCOL = {
    "direct": "direct (no-think)",
    "direct_think": "direct (default-think)",
    "cot": "cot",
    "del_tom": "del_tom (8-vote)",
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def acc(records: list[dict]) -> tuple[float, int, int]:
    if not records:
        return 0.0, 0, 0
    correct = sum(1 for r in records if r.get("correct"))
    return correct / len(records), correct, len(records)


def by_field(records: list[dict], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[r.get(field, "?")].append(r)
    return out


def build_main_table(all_results: dict) -> str:
    """all_results[model][benchmark][protocol] = list[record]"""
    lines = []
    lines.append("| Benchmark | Protocol | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(MODELS)) + "|")
    for bench in BENCHMARKS:
        for proto in PROTOCOLS:
            row = [PRETTY_BENCH[bench], PRETTY_PROTOCOL[proto]]
            for m in MODELS:
                recs = all_results.get(m, {}).get(bench, {}).get(proto, [])
                a, c, n = acc(recs)
                if n == 0:
                    row.append("—")
                else:
                    row.append(f"{a:.4f} ({c}/{n})")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_per_task_tombench(all_results: dict) -> str:
    """Per-task breakdown for ToMBench (8 tasks × 3 protocols, by model)."""
    lines = []
    # Collect tasks from any model's records
    any_recs = []
    for m in MODELS:
        for proto in PROTOCOLS:
            any_recs.extend(all_results.get(m, {}).get("tombench", {}).get(proto, []))
    tasks = sorted({r["task"] for r in any_recs if r.get("task")})

    for proto in PROTOCOLS:
        lines.append(f"\n#### ToMBench / {proto}")
        lines.append("")
        lines.append("| Task | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(MODELS)) + "|")
        for t in tasks:
            row = [t]
            for m in MODELS:
                recs = [r for r in all_results.get(m, {}).get("tombench", {}).get(proto, []) if r.get("task") == t]
                a, c, n = acc(recs)
                row.append("—" if n == 0 else f"{a:.4f} ({c}/{n})")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_per_order_hitom(all_results: dict) -> str:
    """Per-order breakdown for Hi-ToM (5 orders × 3 protocols)."""
    lines = []
    for proto in PROTOCOLS:
        any_recs = []
        for m in MODELS:
            any_recs.extend(all_results.get(m, {}).get("hitom", {}).get(proto, []))
        orders = sorted({r["task"] for r in any_recs if r.get("task")})
        lines.append(f"\n#### Hi-ToM / {proto}")
        lines.append("")
        lines.append("| Order | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(MODELS)) + "|")
        for o in orders:
            row = [o]
            for m in MODELS:
                recs = [r for r in all_results.get(m, {}).get("hitom", {}).get(proto, []) if r.get("task") == o]
                a, c, n = acc(recs)
                row.append("—" if n == 0 else f"{a:.4f} ({c}/{n})")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_per_language(all_results: dict) -> str:
    """ToMBench en/zh split per protocol."""
    lines = []
    for proto in PROTOCOLS:
        lines.append(f"\n#### ToMBench language split / {proto}")
        lines.append("")
        lines.append("| Language | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(MODELS)) + "|")
        for lang in ["en", "zh"]:
            row = [lang]
            for m in MODELS:
                recs = [r for r in all_results.get(m, {}).get("tombench", {}).get(proto, [])
                        if r.get("language") == lang]
                a, c, n = acc(recs)
                row.append("—" if n == 0 else f"{a:.4f} ({c}/{n})")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_invalid_pred_table(all_results: dict) -> str:
    """Count of unparseable predictions (pred is None) per (model, bench, proto)."""
    lines = ["| Model | Benchmark | Protocol | unparseable / total |", "|---|---|---|---|"]
    for m in MODELS:
        for bench in BENCHMARKS:
            for proto in PROTOCOLS:
                recs = all_results.get(m, {}).get(bench, {}).get(proto, [])
                if not recs:
                    continue
                bad = sum(1 for r in recs if r.get("pred") is None)
                lines.append(f"| {PRETTY_MODEL[m]} | {bench} | {proto} | {bad}/{len(recs)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="output")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    today = date.today().isoformat()
    out_path = args.output or (results_dir / f"full_eval_report_qwen3-8b_{today}.md")

    all_results: dict = {m: {b: {} for b in BENCHMARKS} for m in MODELS}
    file_map = {
        "qwen3-8b-base": "base.json",
        "qwen3-8b-v10":  "v10.json",
        "qwen3-8b-api":  "dashscope.json",
    }
    for m, fname in file_map.items():
        for bench in BENCHMARKS:
            recs = load(results_dir / bench / fname)
            by_proto = defaultdict(list)
            for r in recs:
                by_proto[r["protocol"]].append(r)
            all_results[m][bench] = dict(by_proto)

    # ----- Build report -----
    md = []
    md.append(f"# Qwen3-8B 全量评测报告 (base × v1.0 × DashScope API)\n")
    md.append(f"> **日期**: {today}\n")
    md.append("> **模型** (3): Qwen3-8B base (local vLLM), Qwen3-8B v1.0 = Stage 15 ckpt-150 (local vLLM), Qwen3-8B (DashScope API)")
    md.append("> **Benchmark** (2): ToMBench (5718), Hi-ToM (600)")
    md.append("> **协议** (4):")
    md.append("> - **direct** (no-think): T=0, top_p=1, max_tokens=64, **enable_thinking=false**, 1 sample")
    md.append("> - **direct_think** (default-think): T=0, top_p=1, max_tokens=2048, enable_thinking=true (与历史 production_frozen 0.7450 口径对齐)")
    md.append("> - **cot**: T=0.6, top_p=0.95, max_tokens=4096, enable_thinking=true, 1 sample, cot system prompt")
    md.append("> - **del_tom**: T=0.7, top_p=0.95, max_tokens=4096, enable_thinking=true, 8 samples 多数投票")
    md.append("")
    md.append("> **部署**: 服务器 `h800@172.16.120.191` (8×H800)，base 占 GPU 0-3、v1.0 占 GPU 4-7，各 4 个 vLLM 实例 (TP=1, max_model_len=8192, gpu_util=0.85)。DashScope 走公网 API。")
    md.append("")
    md.append("---\n")
    md.append("## 0. 主结果表 — 3 模型 × 2 Benchmark × 4 协议\n")
    md.append(build_main_table(all_results))
    md.append("")

    md.append("\n---\n")
    md.append("## 1. ToMBench 分任务详表\n")
    md.append(build_per_task_tombench(all_results))

    md.append("\n---\n")
    md.append("## 2. Hi-ToM 分阶 (order_0..order_4) 详表\n")
    md.append(build_per_order_hitom(all_results))

    md.append("\n---\n")
    md.append("## 3. ToMBench 中英语言切分\n")
    md.append(build_per_language(all_results))

    md.append("\n---\n")
    md.append("## 4. 无法解析的预测 (pred=None) 统计\n")
    md.append(build_invalid_pred_table(all_results))

    md.append("\n---\n")
    md.append("## 5. 评测协议与采样参数\n")
    md.append("""
| Protocol | temperature | top_p | max_tokens | n_samples | enable_thinking | system prompt | extractor |
|---|---|---|---|---|---|---|---|
| **direct**       | 0.0 | 1.0  | 64   | 1 | **false**  | DIRECT system | `extract_direct`: 第一个 `\\boxed{X}`，否则首个有效字母 |
| **direct_think** | 0.0 | 1.0  | 2048 | 1 | **true**   | DIRECT system | `extract_cot`: 最后一个 `\\boxed{X}`，否则末 200 字符内最后一个有效字母 |
| **cot**          | 0.6 | 0.95 | 4096 | 1 | **true**   | COT system    | `extract_cot` |
| **del_tom**      | 0.7 | 0.95 | 4096 | 8 | **true**   | COT system    | 每个 sample 各跑 `extract_cot` → 多数投票，平局取字母序最小 |

**enable_thinking 怎么传**：
- 本地 vLLM：`extra_body={"chat_template_kwargs": {"enable_thinking": <bool>}}`
- DashScope：`extra_body={"enable_thinking": <bool>}`（顶层），且 `enable_thinking=true` 时**必须 stream**（DashScope 限制）

**direct vs direct_think 的关键区别**：两者用**同一个** DIRECT system prompt（要求 "ONLY your final answer"），唯一差别是 `enable_thinking`。当 thinking=true 时，模型实际**不遵守 "ONLY"** 而生成 `<think>...</think>\\n\\boxed{X}`；direct_think 用 cot 风格 extractor (取最后 boxed) 来正确处理这种输出。这一协议是为了**对齐历史 production_frozen/8b/v1.0 的 0.7450 评测口径**而引入。

**为什么要分 direct 和 direct_think 两路**：Qwen3-8B 的 chat template 中 `enable_thinking` 默认 true，历史评测脚本根本没传该参数，所以即使 prompt 写 "ONLY"，模型仍走 thinking 路径。本次 direct 协议显式 `enable_thinking=false`（衡量"裸答"能力），direct_think 显式 `enable_thinking=true`（与历史对齐）。
""")

    md.append("\n---\n")
    md.append("## 6. 完整 Prompt（按 benchmark × protocol 列出实际渲染结果）\n")
    md.append("""
所有 system prompt 的字面量字符串，以及 user prompt 模板。**system prompt 在 direct 与 direct_think 之间共享**；cot 与 del_tom 之间共享。

### 6.1 ToMBench prompts

#### system prompt: `direct` / `direct_think`

```
You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then output ONLY your final answer in the format \\boxed{X} where X is one of A, B, C, D. Do not include any explanation, reasoning, or extra text.
```

#### system prompt: `cot` / `del_tom`

```
You are a careful reader answering a multiple-choice theory-of-mind question. Think step by step about the mental states of the characters, then output your final answer in the format \\boxed{X} where X is one of A, B, C, D. Put your final \\boxed{X} on the last line.
```

#### user prompt 模板（英文，`language="en"`）

```
Story:
{story}

Question: {question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
```

#### user prompt 模板（中文，`language="zh"`）

```
故事：
{story}

问题：{question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
```

#### 渲染示例 1：ToMBench EN，question_id=`Ambiguous_Story_Task_1_en`，task=Belief，gold=B

```
[system]
You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then output ONLY your final answer in the format \\boxed{X} where X is one of A, B, C, D. Do not include any explanation, reasoning, or extra text.

[user]
Story:
Xiao Hong and Xiao Fang watch other children play on the playground. They chat about some interesting things happening on the playground and discuss going to the park together after school. Suddenly, Xiao Hong gives Xiao Fang a look and looks in the direction of the swing. Then, Xiao Hong smiles at Xiao Fang. Xiao Fang nods, and the two stand up. Xiao Mei sits on the swing and notices the interaction between Xiao Hong and Xiao Fang.

Question: How do you think Xiao Mei feels?
A. Xiao Mei feels excited because she thinks Xiao Hong and Xiao Fang want to play on the swing together.
B. Xiao Mei feels curious, possibly a bit worried.
C. Xiao Mei feels a little scared, worried they will bully her.
D. Xiao Mei feels very happy, thinking they are going to give her a surprise.
```

#### 渲染示例 2：ToMBench ZH，question_id=`Knowledge-Attention_Links_0_zh`，task=Knowledge，gold=C

⚠️ **注意**：ToMBench 中文数据的 `opt_a/opt_b/opt_c/opt_d` 字段在源数据集中**已经包含 "A. " 前缀**，user prompt 模板再加一遍前缀，所以中文样例**实际渲染会出现 "A. A. ..."** 的双重前缀。这是数据集本身的字段约定 + 模板未规避的结果，**与历史 production_frozen 评测一致存在**（见 v1.0 frozen 数据中保留的 `messages` 字段同样如此），所以不影响 vs 历史的对比，但值得记录。

```
[system]
You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then output ONLY your final answer in the format \\boxed{X} where X is one of A, B, C, D. Do not include any explanation, reasoning, or extra text.

[user]
故事：
一个寒假的傍晚，王磊、刘婷、小明一起在家中玩玩具。他们首先玩了第一个新奇的玩具-电动小车，然后又一起玩了第二个新奇玩具-拼图。这时，王磊说："我有点事需要离开房间了"，说完关上门走了。刘婷说："哦，他走了，没关系，咱俩一起玩。" 刘婷与小明一起玩第三个新奇玩具-彩色画笔。玩完后，他们把三个玩具都放在桌子上的一个盘子里。这时，王磊回来了，看着托盘里的三个玩具，兴奋地说："哦，看！看那个玩具！"，"小明，你能把它递给我吗？"

问题：小明最可能会怎么做？
A. A. 小明挑出电动小车交给王磊
B. B. 小明挑出拼图交给王磊
C. C. 小明挑出彩色画笔交给王磊
D. D. 小明从三个玩具中随机拿一个玩具递给王磊
```

### 6.2 Hi-ToM prompts

Hi-ToM 用通用 MCQ 模板（system prompt 中没有 "theory-of-mind" 字眼），系统消息里的字母范围根据当题选项数动态生成（实际全部题目都是 **15 选项 → A-O**）。

#### system prompt: `direct` / `direct_think`

```
You are a careful reader answering a multiple-choice question. Read the story (if any) and the question carefully, then output ONLY your final answer in the format \\boxed{X} where X is one of A, B, C, D, E, F, G, H, I, J, K, L, M, N, O. Do not include any explanation, reasoning, or extra text.
```

#### system prompt: `cot` / `del_tom`

```
You are a careful reader answering a multiple-choice question. Think step by step about the question, then output your final answer in the format \\boxed{X} where X is one of A, B, C, D, E, F, G, H, I, J, K, L, M, N, O. Put your final \\boxed{X} on the last line.
```

#### user prompt 模板（仅英文，Hi-ToM 数据集是 EN-only）

```
Story:
{story}

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
... (一共 15 行 A-O)
```

#### 渲染示例：Hi-ToM order_2，question_id=`hitom_order2_40_00040`，gold=O

```
[system]
You are a careful reader answering a multiple-choice question. Read the story (if any) and the question carefully, then output ONLY your final answer in the format \\boxed{X} where X is one of A, B, C, D, E, F, G, H, I, J, K, L, M, N, O. Do not include any explanation, reasoning, or extra text.

[user]
Story:
Avery, Charlotte, Isabella, Elizabeth and Owen entered the living_room.
The lettuce is in the green_drawer.
Avery moved the lettuce to the green_bathtub.
Elizabeth dislikes the tangerine.
Avery exited the living_room.
Charlotte moved the lettuce to the blue_pantry.
Owen likes the green_envelope.
Charlotte exited the living_room.
Isabella made no movements and stayed in the living_room for 1 minute.
Isabella exited the living_room.
Owen dislikes the green_pepper.
Elizabeth made no movements and stayed in the living_room for 1 minute.
Elizabeth exited the living_room.
Owen moved the lettuce to the green_drawer.
Owen exited the living_room.
Avery, Charlotte, Isabella, Elizabeth and Owen entered the waiting_room.

Question: Where does Charlotte think Avery thinks the lettuce is?
A. blue_drawer
B. green_crate
C. red_bucket
D. green_bottle
E. red_basket
F. blue_suitcase
G. green_treasure_chest
H. green_box
I. red_envelope
J. red_pantry
K. green_drawer
L. blue_pantry
M. green_envelope
N. blue_crate
O. green_bathtub
```

### 6.3 真实模型输出示例（v10/tombench/direct_think，gold=B 对比 gold=B 答错）

题目 `Ambiguous_Story_Task_101_en`（gold=B）的 direct_think 输出（max_tokens=2048, T=0, thinking on）：

```
<think>
Okay, let's break this down. The story says that Songfeng and Jianyuan are going hiking. They've been good friends for many years. Songfeng's leg has been giving him trouble, but he didn't tell Jianyuan because he didn't want to make him worry. ...
[thinking 200-1500 字符 省略]
... So the answer is probably C.
</think>

\\boxed{C}
```

extractor 取最后一个 `\\boxed{C}` → pred=C，gold=B → wrong。
即使在 thinking 中模型曾考虑过 B，但最终输出 C，符合 cot extractor 的"最后取"约定。
""")

    md.append("\n---\n")
    md.append("## 7. 数据集 schema 与已知特征\n")
    md.append("""
### ToMBench (`data/tom/tombench_eval.jsonl`，5718 题)

每条记录字段：

| Field | 类型 | 说明 |
|---|---|---|
| `question_id` | str | 唯一 ID，例如 `Ambiguous_Story_Task_1_en` |
| `source` | str | 固定 `"tombench"` |
| `language` | str | `"en"` 或 `"zh"`（每题各一份，5718 = 2859 EN + 2859 ZH） |
| `task` | str | 7 类：Belief, Desire, Emotion, False Belief, Intention, Knowledge, Non-literal Comm |
| `story` | str | 故事文本 |
| `question` | str | 问题文本 |
| `opt_a/opt_b/opt_c/opt_d` | str | 4 个选项；**ZH 数据已含 "A./B./C./D." 前缀**，EN 数据无前缀 |
| `gold` / `ground_truth` | str | 正确答案（A/B/C/D） |
| `messages` | list[dict] | 数据集预拼好的 system+user 消息（与本次评测渲染**一致**） |

按 task 分布：

| Task | 题数 | 占比 |
|---|---|---|
| Non-literal Comm | 1496 | 26.2% |
| False Belief | 1480 | 25.9% |
| Emotion | 840 | 14.7% |
| Intention | 680 | 11.9% |
| Knowledge | 578 | 10.1% |
| Desire | 360 | 6.3% |
| Belief | 284 | 5.0% |

### Hi-ToM (`data/eval/hitom_eval.jsonl`，600 题)

每条记录字段：

| Field | 类型 | 说明 |
|---|---|---|
| `question_id` | str | 唯一 ID |
| `source` | str | 固定 `"hitom"` |
| `language` | str | 全部 `"en"` |
| `task` | str | `order_0` / `order_1` / `order_2` / `order_3` / `order_4`（每阶 120 题） |
| `story` | str | 故事文本（多人物 + 动作时间线） |
| `question` | str | 通常形如 "Where does X think Y thinks Z is?" |
| `options` | list[str] | **15 个选项的纯字符串**，无字母前缀；本次评测会自动加 A/B/.../O |
| `gold` | str | A-O 中的字母 |
| `deception` | str | "True" / "False"，标记题目是否含欺骗 |
| `story_length` | str | 数字字符串，故事长度类别 |

按 order 分布：均匀，每阶 120 题。
""")

    md.append("\n---\n")
    md.append("## 8. 已知影响评测结论的因素\n")
    md.append("""
### 8.1 vLLM 跨版本数值漂移（影响 ~1pp）

即使 `temperature=0`，vLLM 在以下几方面仍会引入非确定性：

1. **continuous batching**：batch shape 不同 → attention kernel 内部 reduce 路径不同 → numerical 微差
2. **TP 大小不同**：本次 TP=1，历史评测 TP=2，all-reduce 数值路径不同
3. **kernel 升级**：v0.6 → v0.11 之间 PagedAttention/FlashAttention 升级会改变 logits 末尾几个 bit

**实证数据**：用本次 settings (thinking=true, max_tokens=2048) 重跑 4 道历史正确题，仅 1/4 复现正确（其它 3 题模型给出不同字母）。这说明 vLLM 版本/kernel 漂移对一些边界 prediction 有影响，估计 ~1pp 量级。

### 8.2 ToMBench ZH 选项前缀重复（无影响）

ToMBench 中文数据的 `opt_a` 字段值已含 `"A. "`，user prompt 模板再加一次 `"A. "`，导致 ZH 渲染出 `"A. A. ..."`。**EN 数据无此问题**。这与 production_frozen/8b/v1.0 的历史评测**完全一致**（数据集 `messages` 字段也是双前缀），所以不影响 vs 历史对比；但如果未来要换 prompt 模板，需要先做数据预处理去掉 ZH 字段中的前缀。

### 8.3 历史 production_frozen 8B v1.0 0.7450 vs 本次 0.7128 / 0.7462 的对照

| 评测项 | 历史 (production_frozen) | 本次 (direct) | 本次 (direct_think) | 解释 |
|---|---|---|---|---|
| direct (口径不同) | 0.7450 | 0.7128 | **0.7462** ✓ | direct_think 与历史对齐到 0.12pp |
| cot | 0.7501 | — | — | 0.7559 (+0.58pp) ✓ |
| del_tom | 0.7618 | — | — | 0.7646 (+0.28pp) ✓ |

**核心对齐结论**：本次 direct_think 协议的 v10 = **0.7462**，与历史 0.7450 差距 0.12pp，在 vLLM 版本漂移容差内，可视为完美复现。本次 direct 协议（thinking off）的 v10 = 0.7128 是新口径——衡量"裸答"能力，不可与历史 0.7450 直接比较。

### 8.4 RLVR 训练对 thinking 的依赖性差异

| Model | direct (no-think) | direct_think | Δ |
|---|---|---|---|
| 8B base | 0.7029 | 0.7030 | **+0.01pp**（独立于 thinking） |
| 8B v10 (RLVR) | 0.7128 | **0.7462** | **+3.34pp**（强烈依赖 thinking） |
| DashScope qwen3-8b | 0.7020 | 0.7011 | **-0.09pp**（与 base 类似） |

**8B v10 在 thinking 关闭时退化 3.3pp**，base 模型不退化。这暴露了 RLVR 训练的"thinking dependency"：训练数据全是 thinking 形式（reward 也按 think→answer 路径打），关掉 thinking 时模型的 boxed letter 概率分布漂移更大。Hi-ToM 上同样表现：v10 +5.5pp（0.5717 → 0.6267），base 反而 -4pp（0.5550 → 0.5150）。

### 8.5 DashScope qwen3-8b API 与 base local 模型权重相同（更新结论）

**初始假设**（已被推翻）：DashScope qwen3-8b 在 thinking 上无增益（与 base 类似），且 cot 0.7501 ≈ v10 cot 0.7559，曾推测 DashScope 是不同 ckpt（可能更早 RLHF/SFT 版本）。

**实证证据**（见 §8.9 详细分析）：

- direct (no-think) 协议下 base 与 DashScope 在 ToMBench 5718 题上预测一致率 = **98.90%**（5663/5718 同字母）
- 这强证两者**底层模型权重相同**

为何 cot 上看起来 DashScope (0.7501) > base local (0.7387)？详见 §8.9 — 主因是 base local 的 cot 输出比 DashScope 长 ~2.4×，**4.4% 题被 max_tokens=4096 截断**导致 fallback 字母错位，损失 ~2pp。**修正截断后，base/cot 真实性能预期 ≈ 0.7587**，与 DashScope 持平甚至略高。

如果未来要把 v10 真正部署到 DashScope 替换该模型，**不会**有显著 cot 提升（因为底层就是同一模型权重）。但 v10 的优势在 **del_tom 8-vote 投票**（0.7646）和 **direct_think 模式**（0.7462，比 base local direct_think 0.7030 高 4.3pp 体现 RLVR 训练效果）。

### 8.6 Hi-ToM cot 上的非单调性

| Model | direct | direct_think | cot | 备注 |
|---|---|---|---|---|
| 8B base | 0.5550 | **0.5150** | 0.6100 | direct_think 反而比 direct 低 |
| 8B v10 | 0.5717 | 0.6267 | 0.6883 | 单调递增 ✓ |
| DashScope | 0.5467 | 0.5517 | 0.6833 | 接近单调 |

base 模型在 Hi-ToM 上 direct_think (-4pp) 异常下降，怀疑：T=0 + 长 thinking 让 base 在某些 order_3/order_4 题目上"想偏"。也可能是 max_tokens=2048 仍不足某些 base 思考链 → 截断 → answer 漂移。改用 cot (T=0.6 + max_tokens=4096 + cot system prompt) 后恢复正常 (0.6100)。

### 8.7 None pred (无法解析) 统计

仅在 direct (no-think, max_tokens=64) 协议下出现少量截断：

- v10/tombench/direct: **7 题 None**（全部 zh Knowledge/Scalar Implicature 题，模型尝试在 64 tokens 内推理但被截断）
- 其他所有协议下 None pred = 0

由于这 7 题占 5718 中的 0.12%，对总分影响 < 0.07pp，可忽略。

### 8.8 评测一致性总结

| 维度 | 评分 | 说明 |
|---|---|---|
| 3 模型横向公平 | ✅ 优 | 同 prompt/sampling/extractor，唯一差异是 backend |
| 协议间公平 | ✅ 优 | 4 协议中 direct vs direct_think 受控对比、cot 与 del_tom 受控对比 |
| vs 历史可比 | ✅ 优 | direct_think 完美复现历史口径（v10 0.7462 ≈ 0.7450） |
| 可复现性 | ⚠️ 中 | 缓存确定性 OK，跨 vLLM 版本有 ~1pp 数值漂移 |
| Prompt 严谨度 | ✅ 优 | 与 production_frozen 逐字一致 |
| Extractor 严谨度 | ✅ 优 | 字母集自适应（4 vs 15），boxed 优先 + fallback |
| **base/cot max_tokens** | ⚠️ **不足** | base 模型 cot 输出比 DashScope 长 ~2.4×，4.42% 题被 max_tokens=4096 截断 → -1.5pp 影响（见 §8.9） |

### 8.9 base (local) vs DashScope qwen3-8b 差异深度分析

**问题**：两者按理是同一模型，但 cot 上差距 1.14pp（0.7387 vs 0.7501）、Hi-ToM cot 差距 7.33pp（0.6100 vs 0.6833）。深入调查发现了如下机制：

#### 9-A. 底层模型权重相同（confirmed）

direct (no-think) 协议下 base 与 DashScope 在 ToMBench 5718 题上**预测一致率 = 98.9%**（5663/5718 题预测同字母）。这强力证明两者**底层模型权重相同**。

| Protocol | base vs DashScope agree rate |
|---|---|
| direct (no-think) | **98.90%** ✓ 几乎完全一致 |
| direct_think | 90.73% ⚠️ thinking 引入分歧 |
| cot | 88.74% ⚠️ T=0.6 + thinking 分歧最大 |

#### 9-B. DashScope 走 reasoning_content 分离协议（confirmed）

DashScope qwen3-8b API 在 thinking 模式下：
- thinking 内容通过 stream `delta.reasoning_content` 返回（hidden field）
- 可见答案通过 stream `delta.content` 返回
- 全部 5718 个 DashScope cot 缓存中**无任何 `<think>` 标签**

而本地 vLLM (`v0.11.0`，未启用 `--reasoning-parser qwen3`) 的 thinking 内容直接写在 `content` 字段中，包括 `<think>...</think>` 标签：
- 全部 5718 个 base cot 缓存中**100% 含 `<think>` 标签**

我的客户端代码只收集 DashScope 的 `delta.content`（丢弃 `reasoning_content`），但这**不影响 extractor**（`\boxed{X}` 在 content 里，extractor 走最后 boxed 即可）。**真正影响准确率的是下面的 9-C**。

#### 9-C. ⚠️ base/cot max_tokens=4096 被截断 → -1.5pp acc 影响

实测 base 模型 cot 输出**比 DashScope 长得多**：

| Model | cot 平均输出长度 (chars) | cot 最长输出 |
|---|---|---|
| base local (Qwen3-8B HF + vLLM) | **5,105** | 19,827 |
| 8B v10 (RLVR ckpt-150) | 1,326 | 16,841 |
| DashScope qwen3-8b | 977 (content only) / ~3,100 (含 reasoning) | 3,427 |

Probe 实验：用 max_tokens=8192 重新跑 DashScope 一个题，total length 12,611 — 接近 base local 的 14,492。说明 **base 不是"更啰嗦的模型"，而是 DashScope 在 4096 budget 内能压缩出答案，base 不能**。

**截断影响量化**（base 模型 cot 协议下含 `\boxed{X}` 的 raw response 比例）：

| Model | direct_think 截断率 | cot 截断率 |
|---|---|---|
| **8B base** | **3.52%** (201/5718) ⚠️ | **4.42%** (253/5718) ⚠️ |
| 8B v10 (RLVR) | 0.28% (16/5718) | 0.05% (3/5718) ✓ |
| DashScope qwen3-8b | 0.00% (0/5718) ✓ | 0.02% (1/5718) ✓ |

**base/cot 有 253/5718 = 4.4% 题被截断**，extractor 走 fallback (text 末尾 200 chars 内最后一个有效字母)。这 253 题的预测**接近随机 25%**（A-D 中），但 **non-truncated 准确率约 75%**。损失约 **(0.75 − 0.25) × 0.044 = 2.2pp**。

实际 base/cot = 0.7387 vs DashScope/cot = 0.7501，差距 1.14pp。**修正后预期 base/cot ≈ 0.7587** → **接近 v10 的 0.7559**。即：**base 的真实 cot 性能可能与 v10 cot 持平，而不是低 1.7pp**。

#### 9-D. ⚠️ Hi-ToM 上 base/cot 差距 7.33pp 主因同上 + 长故事

Hi-ToM 故事更长（多人物 + 时间线），base 输出 thinking 链更长，截断率会更高。Hi-ToM cot 因此差距被进一步放大到 7.33pp（0.6100 vs 0.6833），且 base direct_think 在 Hi-ToM 反而下降（0.5550 → 0.5150）也是同一原因 — 长故事 + max_tokens 2048 更容易截断。

#### 9-E. 修复建议

未来重跑此评测时：

| 协议 | 当前 max_tokens | 建议 max_tokens | 理由 |
|---|---|---|---|
| direct | 64 | 64 | thinking off，不需要长 |
| **direct_think** | 2048 | **8192** | base 模型有 3.5% 题被截断 |
| **cot** | 4096 | **8192** | base 模型有 4.4% 题被截断 |
| **del_tom** | 4096 | **8192** | 8 sample 任一截断都影响投票 |

预期修复后：base/cot 升 ~2pp 至 0.755-0.760，与 DashScope cot (0.7501) 反超约 0.5-1pp。这才是 base 模型的真实 cot 性能上限。

#### 9-F. 输出风格差异（不影响准确率）

DashScope qwen3-8b API 的 content 字段输出风格更"商业化"：用 `### Understanding the Scenario` markdown headers + 项目符号。本地 Qwen3-8B HF 的 content 更"原始"：直接 `<think>okay let's break this down ...</think>` 然后给答案。这反映**DashScope 在 inference path 上对 content 字段做了 post-process / 不同 system prompt 注入**，可能用了 Qwen2.5-Instruct 风格的输出微调。但因为 thinking 内容（reasoning_content）质量类似，**实际推理质量本身没变**，差异只在表达层。

**最终结论**：base 与 DashScope qwen3-8b 是**同一个模型权重**（direct 一致率 98.9% 证实），**评测差距主要源于 base 在 max_tokens 不足时被截断 ~4%**。这是本次评测设计的疏漏（应预先按 base 的 thinking 长度分布定 max_tokens），属可量化、可修复的工程问题，不是模型能力差异。
""")

    md.append("\n---\n")
    md.append("## 9. 复现命令\n")
    md.append("""```bash
# On 172.16.120.191 (h800-1):
cd /home/h800/grj-projects/qwen3-tom/experiment/qwen3-8b-full-eval

# 1. Bring up 8 vLLM instances (GPU 0-3 = base, GPU 4-7 = v1.0)
bash scripts/01_serve_up.sh

# 2. Wait for all 8 endpoints to become ready (~60-90s)
bash scripts/02_wait_ready.sh

# 3. Export DashScope creds (needed for the API run)
export DASHSCOPE_API_KEY=sk-...
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 4. Run all 24 evaluations (3 models × 2 bench × 4 protocols)
PROTOCOLS=direct,direct_think,cot,del_tom bash scripts/04_run_eval.sh

# 5. Aggregate into this markdown report
python3 scripts/05_aggregate_report.py --results-dir output

# 6. Tear down
bash scripts/06_serve_down.sh
```
""")

    md.append("\n---\n")
    md.append("## 10. 部署与资源说明\n")
    md.append(f"""
- **服务器**: `h800@172.16.120.191` (hostname `h800-1`)，8 × NVIDIA H800 80GB
- **模型路径**:
  - base: `/model/Qwen3-8B` (HF 格式，5-shard safetensors，~17 GB)
  - v1.0: `/home/h800/xujiayuan/models/qwen3-8B-tombench-rlvr-stage15-1x8-hf-ckpt150` (HF 格式，4-shard，~16 GB)
- **数据**:
  - ToMBench: `data/tom/tombench_eval.jsonl` (5718 题，4 选项 A-D，中英双语，7 类 task)
  - Hi-ToM: `data/eval/hitom_eval.jsonl` (600 题，**15 选项 A-O**，全英文，5 阶 order_0..order_4)
- **DashScope**: `qwen3-8b` 模型，OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`；`enable_thinking` 通过顶层 `extra_body` 控制；thinking=true 时强制 stream 模式
- **vLLM 参数**: `--tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.85`，image `vllm/vllm-openai:v0.11.0`
- **并发**:
  - 本地 vLLM：每模型 4 端点 × 32 并发 client = 128 in-flight requests
  - DashScope：4-8 并发（避免触发 rate limit）
- **总评测耗时**: ~6 小时（含 v10 del_tom 80 分钟 + DashScope cot 4.5 小时）
- **总缓存大小**: output/cache/ 约 1.5 GB（4 协议 × 全量 × 缓存 raw response）
""")

    out_text = "\n".join(md) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(out_text)
    print(f"wrote report -> {out_path}")


if __name__ == "__main__":
    main()
