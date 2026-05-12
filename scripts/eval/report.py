"""Aggregate per-question eval results into model-x-protocol-x-language-x-task table."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable


def aggregate_results(records: Iterable[dict]) -> dict:
    """Return dict {(model, protocol): {overall, en, zh, task: {<task>: acc}}}.

    Each record requires: model, protocol, language, task, correct (bool).
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["model"], r["protocol"])].append(r)

    result = {}
    for key, rs in by_key.items():
        n = len(rs)
        correct = sum(1 for r in rs if r["correct"])
        overall = correct / n if n else 0.0

        en_rs = [r for r in rs if r["language"] == "en"]
        zh_rs = [r for r in rs if r["language"] == "zh"]

        en_acc = sum(1 for r in en_rs if r["correct"]) / len(en_rs) if en_rs else 0.0
        zh_acc = sum(1 for r in zh_rs if r["correct"]) / len(zh_rs) if zh_rs else 0.0

        task_acc: dict[str, float] = {}
        task_groups: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            task_groups[r["task"]].append(r)
        for task, t_rs in task_groups.items():
            task_acc[task] = sum(1 for r in t_rs if r["correct"]) / len(t_rs)

        result[key] = {
            "overall": overall,
            "en": en_acc,
            "zh": zh_acc,
            "task": task_acc,
            "n": n,
        }
    return result


def format_markdown_table(agg: dict) -> str:
    """Format the main results table."""
    lines = ["| Model | Protocol | n | Overall | EN | ZH |",
             "|---|---|---|---|---|---|"]
    for (model, protocol), cell in sorted(agg.items()):
        n_val = cell.get("n", "-")
        lines.append(
            f"| {model} | {protocol} | {n_val} | "
            f"{cell['overall']:.4f} | {cell['en']:.4f} | {cell['zh']:.4f} |"
        )
    lines.append("")
    # Per-task breakdown
    lines.append("## Per-task breakdown")
    all_tasks: set[str] = set()
    for cell in agg.values():
        all_tasks.update(cell["task"].keys())
    sorted_tasks = sorted(all_tasks)
    header = "| Model | Protocol | " + " | ".join(sorted_tasks) + " |"
    sep = "|" + "---|" * (2 + len(sorted_tasks))
    lines.extend([header, sep])
    for (model, protocol), cell in sorted(agg.items()):
        row = f"| {model} | {protocol} |"
        for t in sorted_tasks:
            v = cell["task"].get(t)
            row += f" {v:.4f} |" if v is not None else " - |"
        lines.append(row)
    return "\n".join(lines)
