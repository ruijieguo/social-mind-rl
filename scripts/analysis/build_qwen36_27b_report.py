#!/usr/bin/env python3
# Aggregate the 4 ToM-family benchmark JSONs produced by
# scripts/eval/run_qwen36_27b_baseline.sh into a single markdown report
# matching the layout of output/eval/baseline_report.md.
#
# Usage:
#   python scripts/analysis/build_qwen36_27b_report.py \
#     --input-dir output/eval/qwen36_27b_baseline \
#     --output    output/eval/qwen36_27b_baseline_report.md
#
# Optional comparators (any subset):
#   --compare-stage16 output/eval/stage16_ckpt270_{tombench,emobench,hitom,socialiqa}.json
#   --compare-deepseek output/eval/{deepseek_*}.json

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def acc(rows: Iterable[dict]) -> tuple[float, int]:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return float("nan"), 0
    return sum(1 for r in rows if r.get("correct")) / n, n


def fmt(a: float, n: int) -> str:
    if n == 0:
        return "—"
    return f"{a:.4f} (n={n})"


def by(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def section_overall(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"## {name}\n\n_(no data)_\n\n"
    lines = [f"## {name}", "", "| Model | Protocol | n | Overall |", "|---|---|---|---|"]
    for (model, proto), grp in sorted(by(rows, "model", "protocol").items()):
        a, n = acc(grp)
        lines.append(f"| {model} | {proto} | {n} | {a:.4f} |")
    return "\n".join(lines) + "\n\n"


def section_by_task(name: str, rows: list[dict], task_key: str = "task") -> str:
    if not rows:
        return ""
    tasks = sorted({r.get(task_key) for r in rows if r.get(task_key) is not None})
    if not tasks:
        return ""
    head = ["Model", "Protocol"] + list(tasks)
    sep = ["---"] * len(head)
    lines = [f"### {name} — by {task_key}", "", "| " + " | ".join(head) + " |",
             "| " + " | ".join(sep) + " |"]
    for (model, proto), grp in sorted(by(rows, "model", "protocol").items()):
        per_task = by(grp, task_key)
        cells = [model, proto]
        for t in tasks:
            a, n = acc(per_task.get((t,), []))
            cells.append(f"{a:.4f}" if n else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n"


def section_by_lang(name: str, rows: list[dict]) -> str:
    langs = sorted({r.get("language") for r in rows if r.get("language")})
    if len(langs) < 2:
        return ""
    head = ["Model", "Protocol", "n", "Overall"] + langs
    sep = ["---"] * len(head)
    lines = [f"### {name} — by language", "", "| " + " | ".join(head) + " |",
             "| " + " | ".join(sep) + " |"]
    for (model, proto), grp in sorted(by(rows, "model", "protocol").items()):
        oa, on = acc(grp)
        cells = [model, proto, str(on), f"{oa:.4f}"]
        per_lang = by(grp, "language")
        for ln in langs:
            a, n = acc(per_lang.get((ln,), []))
            cells.append(f"{a:.4f}" if n else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n"


BENCH_FILES = {
    "ToMBench": "tombench.json",
    "EmoBench": "emobench.json",
    "Hi-ToM": "hitom.json",
    "SocialIQA": "socialiqa.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--compare-stage16-dir", type=Path,
                    help="Dir containing stage16_ckpt270_*.json to fold into headline table")
    ap.add_argument("--compare-deepseek-dir", type=Path,
                    help="Dir containing deepseek_*.json")
    args = ap.parse_args()

    benches: dict[str, list[dict]] = {}
    for label, fname in BENCH_FILES.items():
        benches[label] = load(args.input_dir / fname)

    # Headline table: model × bench × protocol overall accuracy.
    headline_rows: list[tuple[str, str, str, float, int]] = []  # (model, bench, proto, acc, n)

    def collect(label: str, rows: list[dict]) -> None:
        for (model, proto), grp in sorted(by(rows, "model", "protocol").items()):
            a, n = acc(grp)
            headline_rows.append((model, label, proto, a, n))

    for label, rows in benches.items():
        collect(label, rows)

    # Optional comparators (read from disk if dirs given).
    if args.compare_stage16_dir:
        for label, fname in {
            "ToMBench": "stage16_ckpt270_tombench.json",
            "EmoBench": "stage16_ckpt270_emobench.json",
            "Hi-ToM": "stage16_ckpt270_hitom.json",
            "SocialIQA": "stage16_ckpt270_socialiqa.json",
        }.items():
            collect(label, load(args.compare_stage16_dir / fname))
    if args.compare_deepseek_dir:
        for label, fname in {
            "EmoBench": "deepseek_emobench_full.json",
            "Hi-ToM": "deepseek_hitom_direct_only.json",
            "SocialIQA": "deepseek_socialiqa_full.json",
        }.items():
            collect(label, load(args.compare_deepseek_dir / fname))

    out: list[str] = []
    out.append("# Qwen3.6-27B baseline — 4 ToM-family benchmarks\n")
    out.append("Protocols: `direct`, `cot`, `del_tom (n=8 votes)`. "
               "All rows are accuracy unless stated otherwise.\n\n")

    # Headline table.
    out.append("## Headline\n")
    out.append("| Model | Benchmark | Protocol | n | Acc |\n|---|---|---|---|---|\n")
    for model, bench, proto, a, n in sorted(headline_rows):
        out.append(f"| {model} | {bench} | {proto} | {n} | {a:.4f} |\n")
    out.append("\n")

    # Per-benchmark sections.
    for label, rows in benches.items():
        out.append(section_overall(label, rows))
        if label == "ToMBench":
            out.append(section_by_task("ToMBench", rows, "task"))
            out.append(section_by_lang("ToMBench", rows))
        elif label == "EmoBench":
            out.append(section_by_task("EmoBench", rows, "task"))
            out.append(section_by_lang("EmoBench", rows))
        elif label == "Hi-ToM":
            out.append(section_by_task("Hi-ToM", rows, "task"))
        elif label == "SocialIQA":
            pass  # single task, no breakdown

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(out))
    print(f"wrote {args.output}  ({sum(len(r) for r in benches.values())} total rows)")


if __name__ == "__main__":
    main()
