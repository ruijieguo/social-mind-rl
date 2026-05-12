"""Compare two eval result JSONs (e.g., baseline vs trained) into markdown."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.eval.report import aggregate_results, format_markdown_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="output/eval/baseline_combined.json")
    p.add_argument("--trained", default="output/eval/final.json")
    p.add_argument("--out", default="output/analysis/eval_diff.md")
    args = p.parse_args()

    base_records = json.loads(Path(args.baseline).read_text()) if Path(args.baseline).exists() else []
    train_records = json.loads(Path(args.trained).read_text()) if Path(args.trained).exists() else []

    base_agg = aggregate_results(base_records)
    train_agg = aggregate_results(train_records)

    lines = ["# Eval diff: baseline vs trained", ""]

    # Combined main table
    combined: dict = {}
    for k, v in base_agg.items():
        combined[k] = v
    for k, v in train_agg.items():
        combined[k] = v
    lines.append(format_markdown_table(combined))

    # Compute deltas vs deepseek-v4-pro X
    direct_keys = [k for k in combined if k[1] == "direct"]
    x = next((combined[k]["overall"] for k in direct_keys if "deepseek" in k[0]), None)
    if x is not None:
        lines.append("")
        lines.append("## Distance to deepseek-v4-pro (X) on direct overall")
        lines.append("| Model | overall | X − overall | meets ε=0.02? |")
        lines.append("|---|---|---|---|")
        for k in sorted(direct_keys):
            overall = combined[k]["overall"]
            delta = x - overall
            status = "✓ (or better)" if delta <= 0.02 else "✗"
            lines.append(f"| {k[0]} | {overall:.4f} | {delta:+.4f} | {status} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
