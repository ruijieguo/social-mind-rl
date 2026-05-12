"""Re-aggregate baseline_combined.json with the latest task mapping."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import jsonlines

from scripts.eval.report import aggregate_results, format_markdown_table


def main():
    # Build qid -> updated_task map from the freshly-rebuilt eval JSONL
    qid_to_task: dict[str, str] = {}
    with jsonlines.open("data/tom/tombench_eval.jsonl") as r:
        for row in r:
            qid_to_task[row["question_id"]] = row["task"]

    src = Path("output/eval/baseline_combined.json")
    records = json.loads(src.read_text())
    n_updated = 0
    for r in records:
        new_task = qid_to_task.get(r["question_id"])
        if new_task and new_task != r.get("task"):
            r["task"] = new_task
            n_updated += 1

    print(f"Updated {n_updated}/{len(records)} records with refreshed task mapping",
          file=sys.stderr, flush=True)

    # Persist the updated records back
    src.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    agg = aggregate_results(records)
    md_path = Path("output/eval/baseline_report.md")
    md_path.write_text(format_markdown_table(agg))
    print(f"Wrote {md_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
