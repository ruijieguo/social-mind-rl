"""Audit incorrect answers from the trained model: 5 sample errors per task."""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="output/eval/final.json")
    p.add_argument("--out", default="output/analysis/errors.md")
    p.add_argument("--per-task", type=int, default=5)
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"results file {results_path} not found; writing empty audit")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Error audit\n\n_(no results yet)_\n")
        return

    records = json.loads(results_path.read_text())
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not r["correct"] and r.get("protocol") == "direct":
            by_task[r["task"]].append(r)

    lines = ["# Error audit (direct protocol, trained model)", ""]
    random.seed(42)
    for task, errs in sorted(by_task.items()):
        lines.append(f"## {task} — {len(errs)} errors total")
        lines.append("")
        sample = random.sample(errs, k=min(args.per_task, len(errs)))
        for r in sample:
            lines.append(f"### qid: {r['question_id']} ({r['language']})")
            lines.append(f"- gold: **{r['gold']}**, pred: **{r['pred']}**")
            resp = (r.get("raw_responses") or [""])[0]
            resp = resp[:500] + "..." if len(resp) > 500 else resp
            lines.append("- raw response:")
            lines.append(f"  ```\n  {resp}\n  ```")
            lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
