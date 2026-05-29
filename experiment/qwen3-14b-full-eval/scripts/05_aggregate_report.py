"""Aggregate Qwen3-14B full-eval results into a markdown report.

Reads  output/{tombench,hitom,socialiqa,emobench}/{base,v35,v31}.json
Writes output/full_eval_report_qwen3-14b_<DATE>.md

Covers: main accuracy table, per-task / per-order / language breakdowns,
truncation analysis (finish_reason == "length"), unparseable preds, and output
length stats.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

MODELS = ["base", "v35", "v31", "deepseek"]
PRETTY_MODEL = {
    "base": "Qwen3-14B base",
    "v35":  "v3.5 (Stage 19 ckpt-120)",
    "v31":  "v3.1 (Stage 14b ckpt-199)",
    "deepseek": "deepseek-v4-pro (API)",
}
BENCHMARKS = ["tombench", "emobench", "socialiqa", "hitom"]
PRETTY_BENCH = {
    "tombench":  "ToMBench (n=5718)",
    "emobench":  "EmoBench (n=1200)",
    "socialiqa": "SocialIQA (n=1954)",
    "hitom":     "Hi-ToM (n=600)",
}
PROTOCOLS = ["direct", "direct_think", "cot"]
PRETTY_PROTO = {
    "direct": "direct (no-think)",
    "direct_think": "direct (default-think)",
    "cot": "cot",
}


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def acc(recs):
    if not recs:
        return 0.0, 0, 0
    c = sum(1 for r in recs if r.get("correct"))
    return c / len(recs), c, len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="output")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    today = date.today().isoformat()
    out_path = args.output or (rd / f"full_eval_report_qwen3-14b_{today}.md")

    # data[model][bench][proto] = list[record]
    data: dict = {m: {b: defaultdict(list) for b in BENCHMARKS} for m in MODELS}
    for m in MODELS:
        for b in BENCHMARKS:
            for r in load(rd / b / f"{m}.json"):
                data[m][b][r["protocol"]].append(r)

    def cell(m, b, p):
        return data[m][b].get(p, [])

    md: list[str] = []
    A = md.append

    A(f"# Qwen3-14B 全量评测报告 — base × v3.5 × v3.1 × deepseek-v4-pro\n")
    A(f"> **日期**: {today}\n")
    A("> **模型** (4):")
    A(">")
    A("> - **base** = 原始 Qwen3-14B（ModelScope 下载，本地 vLLM）")
    A("> - **v3.5** = Stage 19 ckpt-120（GPT-5.5 蒸馏改进版，本地 vLLM）")
    A("> - **v3.1** = Stage 14b ckpt-199（task-weighted 重采样，历史 ToMBench del_tom 最高，本地 vLLM）")
    A("> - **deepseek-v4-pro** = DeepSeek 官方 API（参照目标 X）")
    A(">")
    A("> **Benchmark** (4): ToMBench 5718 / EmoBench 1200 / SocialIQA 1954 / Hi-ToM 600")
    A("> **协议** (3):")
    A("> - **direct** (no-think): T=0, top_p=1, max_tokens=64, enable_thinking=false")
    A("> - **direct_think** (default-think): T=0, top_p=1, **max_tokens=8192**, enable_thinking=true")
    A("> - **cot**: T=0.6, top_p=0.95, **max_tokens=8192**, enable_thinking=true")
    A(">")
    A("> **部署**: `h800@172.16.120.181` (8×H800)，GPUs 4-7 各 1 个 vLLM 实例 (TP=1, "
      "max_model_len=16384, gpu_util=0.85)，4 端点 round-robin。GPUs 0-3 为他人作业，未占用。")
    A("\n---\n")

    # ---- 0. main table -----------------------------------------------------
    A("## 0. 主结果表\n")
    A("| Benchmark | Protocol | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
    A("|---|---|" + "|".join(["---"] * len(MODELS)) + "|")
    for b in BENCHMARKS:
        for p in PROTOCOLS:
            row = [PRETTY_BENCH[b], PRETTY_PROTO[p]]
            for m in MODELS:
                a, c, n = acc(cell(m, b, p))
                row.append("—" if n == 0 else f"{a:.4f}")
            A("| " + " | ".join(row) + " |")
    A("")

    # ---- 0.5 best-protocol summary ----------------------------------------
    A("\n---\n")
    A("## 0.5 每模型每 benchmark 最优协议 + 均值\n")
    A("| Model | " + " | ".join(PRETTY_BENCH[b] for b in BENCHMARKS) + " | 均值 |")
    A("|---|" + "|".join(["---"] * (len(BENCHMARKS) + 1)) + "|")
    for m in MODELS:
        cells, vals = [], []
        for b in BENCHMARKS:
            best = None
            for p in PROTOCOLS:
                a, c, n = acc(cell(m, b, p))
                if n and (best is None or a > best[0]):
                    best = (a, p)
            if best:
                cells.append(f"{best[0]:.4f} ({best[1][:3]})"); vals.append(best[0])
            else:
                cells.append("—")
        avg = sum(vals) / len(vals) if vals else 0.0
        A("| " + PRETTY_MODEL[m] + " | " + " | ".join(cells) + f" | {avg:.4f} |")
    A("")

    # ---- 1. truncation analysis (the requested check) ---------------------
    A("\n---\n")
    A("## 1. 截断检测 (finish_reason == \"length\")\n")
    A("评测期间逐条记录 vLLM 的 `finish_reason`。`length` = 输出撑满 max_tokens 被截断，"
      "thinking 协议下会导致 `\\boxed{}` 丢失 → extractor fallback → 准确率虚低（8B v1 的已知坑）。\n")
    A("| Model | Benchmark | Protocol | truncated / total | 截断率 |")
    A("|---|---|---|---|---|")
    any_trunc = False
    for m in MODELS:
        for b in BENCHMARKS:
            for p in PROTOCOLS:
                recs = cell(m, b, p)
                if not recs:
                    continue
                t = sum(1 for r in recs if r.get("truncated"))
                if t:
                    any_trunc = True
                flag = " ⚠️" if t / len(recs) >= 0.01 else ""
                A(f"| {PRETTY_MODEL[m]} | {b} | {p} | {t}/{len(recs)} | {t/len(recs):.2%}{flag} |")
    if not any_trunc:
        A("\n**✅ 全部 cell 截断率为 0 —— max_tokens=8192 充足，无截断问题。**")
    A("")

    # ---- 1.5 output length stats ------------------------------------------
    A("\n---\n")
    A("## 1.5 输出长度统计 (chars)\n")
    A("| Model | Benchmark | Protocol | mean | p95 | max |")
    A("|---|---|---|---|---|---|")
    for m in MODELS:
        for b in BENCHMARKS:
            for p in PROTOCOLS:
                recs = cell(m, b, p)
                if not recs:
                    continue
                lens = sorted(r.get("out_len", 0) for r in recs)
                mean = sum(lens) / len(lens)
                p95 = lens[int(0.95 * (len(lens) - 1))]
                A(f"| {PRETTY_MODEL[m]} | {b} | {p} | {mean:.0f} | {p95} | {lens[-1]} |")
    A("")

    # ---- 2. unparseable preds ---------------------------------------------
    A("\n---\n")
    A("## 2. 无法解析的预测 (pred=None) 统计\n")
    A("| Model | Benchmark | Protocol | unparseable / total |")
    A("|---|---|---|---|")
    for m in MODELS:
        for b in BENCHMARKS:
            for p in PROTOCOLS:
                recs = cell(m, b, p)
                if not recs:
                    continue
                bad = sum(1 for r in recs if r.get("pred") is None)
                A(f"| {PRETTY_MODEL[m]} | {b} | {p} | {bad}/{len(recs)} |")
    A("")

    # ---- 3. per-task / per-order breakdowns -------------------------------
    def breakdown(bench, field, label):
        A(f"\n---\n")
        A(f"## {label}\n")
        keys = sorted({r.get(field) for m in MODELS for p in PROTOCOLS
                       for r in cell(m, bench, p) if r.get(field)})
        for p in PROTOCOLS:
            A(f"\n#### {bench} / {p}")
            A("")
            A(f"| {field} | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
            A("|---|" + "|".join(["---"] * len(MODELS)) + "|")
            for k in keys:
                row = [str(k)]
                for m in MODELS:
                    recs = [r for r in cell(m, bench, p) if r.get(field) == k]
                    a, c, n = acc(recs)
                    row.append("—" if n == 0 else f"{a:.4f} ({c}/{n})")
                A("| " + " | ".join(row) + " |")

    breakdown("tombench", "task", "3. ToMBench 分任务详表")
    breakdown("hitom", "task", "4. Hi-ToM 分阶 (order_0..order_4)")
    breakdown("emobench", "task", "5. EmoBench 分任务 (EA/EU)")

    # ToMBench language split
    A("\n---\n")
    A("## 6. ToMBench 中英语言切分\n")
    for p in PROTOCOLS:
        A(f"\n#### ToMBench language split / {p}")
        A("")
        A("| Language | " + " | ".join(PRETTY_MODEL[m] for m in MODELS) + " |")
        A("|---|" + "|".join(["---"] * len(MODELS)) + "|")
        for lang in ["en", "zh"]:
            row = [lang]
            for m in MODELS:
                recs = [r for r in cell(m, "tombench", p) if r.get("language") == lang]
                a, c, n = acc(recs)
                row.append("—" if n == 0 else f"{a:.4f} ({c}/{n})")
            A("| " + " | ".join(row) + " |")

    # ---- 7. protocol params + deploy --------------------------------------
    A("\n---\n")
    A("## 7. 评测协议与采样参数\n")
    A("""
| Protocol | temperature | top_p | max_tokens | enable_thinking | system prompt | extractor |
|---|---|---|---|---|---|---|
| **direct**       | 0.0 | 1.0  | 64   | **false** | DIRECT system | `extract_direct`: 第一个 `\\boxed{X}`，否则首个有效字母 |
| **direct_think** | 0.0 | 1.0  | 8192 | **true**  | DIRECT system | `extract_cot`: 最后一个 `\\boxed{X}`，否则末 200 字符内最后一个有效字母 |
| **cot**          | 0.6 | 0.95 | 8192 | **true**  | COT system    | `extract_cot` |

- 本地 vLLM（base/v3.5/v3.1）通过 `extra_body={"chat_template_kwargs": {"enable_thinking": <bool>}}` 控制 thinking。
- **deepseek-v4-pro**（DeepSeek 官方 API）：所有采样参数（T / top_p / max_tokens）、prompt、extractor 与本地模型**完全一致**，
  以保证公平。`enable_thinking` 是 Qwen/vLLM 专有开关，deepseek 为原生推理模型、无此旋钮，故不传；
  其推理走独立的 `reasoning_content` 字段，可见答案（含 `\\boxed{}`）在 `content` 中，我们与本地模型一样从 `content` 抽取。
- ⚠️ **deepseek-v4-pro 的 `direct`（max_tokens=64）几乎必然截断**：推理模型在 64 token 内来不及给出答案，
  reasoning 就耗尽预算、`content` 为空 → pred=None。这是"完全一致参数"作用在推理模型上的必然结果，
  **deepseek 的有效对比应看 `direct_think` / `cot`**（见 §1 截断表）。
- ToMBench 用 ToM 专用 system prompt（含 "theory-of-mind"，4 选项 A-D，ZH 选项自动 strip 重复字母前缀）。
- Hi-ToM / SocialIQA / EmoBench 用通用 MCQ system prompt，字母范围按当题选项数动态生成
  （Hi-ToM 15→A-O，EmoBench 4→A-D，SocialIQA 3→A-C），与 `scripts/eval/run_generic_mcq.py` 一致。
""")
    A("\n---\n")
    A("## 8. 部署与复现\n")
    A("""```bash
# On 172.16.120.181 (h800-3):
cd /data_nvme/grj-projects/qwen3-tom/experiment/qwen3-14b-full-eval

# Full run (per-model: up GPUs 4-7 → eval 4 benches × 3 protocols → down)
bash scripts/04_run_eval.sh

# Smoke test (10 questions)
LIMIT=10 bash scripts/04_run_eval.sh

# Aggregate this report
python3 scripts/05_aggregate_report.py --results-dir output
```

- **模型路径**:
  - base: `/data_nvme/grj-projects/models/Qwen3-14B`
  - v3.5: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage19-1x8-hf-ckpt120`
  - v3.1: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage14b-1x8-hf-ckpt199`
- **数据**: `data/tom/tombench_eval.jsonl`, `data/eval/{hitom,socialiqa,emobench}_eval.jsonl`
- **vLLM**: image `qwen3-tom-serve-eval-dp4:latest`, TP=1, max_model_len=16384, gpu_util=0.85
""")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(md) + "\n")
    print(f"wrote report -> {out_path}")


if __name__ == "__main__":
    main()
