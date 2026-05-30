"""Summarize the base⊕v3.1 model-soup alpha sweep.

Prints the alpha curve: for each model (base=α0, soup25, soup50, soup75, v31=α1,
and deepseek for reference), the best-protocol accuracy per benchmark + the
4-benchmark mean. Identifies the alpha that maximizes the mean.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# (model_id, alpha-or-label)
ROW_ORDER = [
    ("base", "α=0.00 (base)"),
    ("soup25", "α=0.25"),
    ("soup50", "α=0.50"),
    ("soup75", "α=0.75"),
    ("v31", "α=1.00 (v3.1)"),
    ("deepseek", "deepseek-v4-pro"),
]
BENCHES = ["tombench", "emobench", "socialiqa", "hitom"]
PROTOCOLS = ["direct", "direct_think", "cot"]


def load(rd, b, m):
    f = rd / b / f"{m}.json"
    return json.loads(f.read_text()) if f.exists() else []


def best_proto(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["protocol"]].append(r)
    best = None
    for p in PROTOCOLS:
        if by.get(p):
            a = sum(x["correct"] for x in by[p]) / len(by[p])
            if best is None or a > best[0]:
                best = (a, p)
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="output")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    print(f"{'model':18s}" + "".join(f"{b:>12s}" for b in BENCHES) + f"{'mean':>9s}")
    print("-" * 78)
    rows = []
    for mid, label in ROW_ORDER:
        cells, vals = [], []
        for b in BENCHES:
            recs = load(rd, b, mid)
            bp = best_proto(recs)
            if bp:
                cells.append(f"{bp[0]:.4f}({bp[1][:3]})")
                vals.append(bp[0])
            else:
                cells.append("    --    ")
        mean = sum(vals) / len(vals) if len(vals) == len(BENCHES) else float("nan")
        rows.append((label, mean))
        print(f"{label:18s}" + "".join(f"{c:>12s}" for c in cells) + f"{mean:9.4f}")

    # winner among soups+endpoints (exclude deepseek)
    local = [(l, m) for l, m in rows if "deepseek" not in l and m == m]  # m==m drops NaN
    if local:
        best = max(local, key=lambda x: x[1])
        print("-" * 78)
        print(f"BEST local mean: {best[0]} → {best[1]:.4f}")


if __name__ == "__main__":
    main()
