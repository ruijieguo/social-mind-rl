"""Aggregate Stage 22 post-training eval: 4-bench best-protocol mean per model,
compared against base (measured here) and v3.1 (cited from the 2026-05-30 eval).

Prints the alpha-style table + the Plan-A verdict:
  success = a ckpt with 4-bench mean > base 0.7603 AND Hi-ToM cot not collapsed.
"""
from __future__ import annotations
import argparse, json, os, collections

BENCHES = ["tombench", "emobench", "socialiqa", "hitom"]
PROTOCOLS = ["direct", "direct_think", "cot"]
# cited from experiment/qwen3-14b-full-eval (same harness, 4-bench best-protocol mean)
CITED = {"v3.1 (cited)": {"mean": 0.7305, "tombench": 0.7816, "emobench": 0.6483,
                          "socialiqa": 0.7886, "hitom": 0.7033}}
BASE_MEAN_PRIOR = 0.7603


def load(rd, b, m):
    f = os.path.join(rd, b, f"{m}.json")
    return json.load(open(f)) if os.path.exists(f) else []


def best_proto(recs):
    by = collections.defaultdict(list)
    for r in recs:
        by[r["protocol"]].append(r)
    best = None
    for p in PROTOCOLS:
        if by.get(p):
            a = sum(x["correct"] for x in by[p]) / len(by[p])
            if best is None or a > best[0]:
                best = (a, p)
    return best


def cot_acc(recs):
    c = [r for r in recs if r["protocol"] == "cot"]
    return (sum(x["correct"] for x in c) / len(c)) if c else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="output")
    ap.add_argument("--ckpts", default="50 100 150 200")
    args = ap.parse_args()
    rd = args.results_dir
    models = ["base"] + [f"ckpt{n}" for n in args.ckpts.split()]

    print(f"\n{'model':16s}" + "".join(f"{b:>11s}" for b in BENCHES) + f"{'mean':>9s}{'hitom_cot':>11s}")
    print("-" * 90)
    rows = {}
    base_mean = None
    for m in models:
        cells, vals = [], []
        hitom_cot = float("nan")
        for b in BENCHES:
            recs = load(rd, b, m)
            bp = best_proto(recs)
            if bp:
                cells.append(f"{bp[0]:.4f}({bp[1][:3]})"); vals.append(bp[0])
            else:
                cells.append("   --   ")
            if b == "hitom":
                hitom_cot = cot_acc(recs)
        mean = sum(vals) / len(vals) if len(vals) == 4 else float("nan")
        rows[m] = (mean, hitom_cot)
        if m == "base":
            base_mean = mean
        label = {"base": "base (measured)"}.get(m, m)
        print(f"{label:16s}" + "".join(f"{c:>11s}" for c in cells)
              + f"{mean:9.4f}{hitom_cot:11.4f}")
    for lbl, d in CITED.items():
        print(f"{lbl:16s}" + "".join(f"{d[b]:11.4f}" for b in BENCHES)
              + f"{d['mean']:9.4f}{d['hitom']:11.4f}")

    print("-" * 90)
    anchor = base_mean if base_mean == base_mean else BASE_MEAN_PRIOR
    print(f"base mean (anchor): {anchor:.4f}   (prior full-eval base = {BASE_MEAN_PRIOR})")
    ck = {m: v for m, v in rows.items() if m != "base" and v[0] == v[0]}
    if ck:
        best = max(ck.items(), key=lambda kv: kv[1][0])
        bm, bh = best[1]
        print(f"BEST ckpt by 4-bench mean: {best[0]} → mean {bm:.4f}, Hi-ToM cot {bh:.4f}")
        beats = bm > anchor
        keeps = bh >= 0.76
        verdict = ("✅ PLAN A SUCCESS" if beats and keeps else
                   "🟡 beats base but Hi-ToM weak" if beats else
                   "🟡 Hi-ToM held but mean ≤ base" if keeps else
                   "🔴 PLAN A did NOT beat base")
        print(f"VERDICT: {verdict}  (beats base mean: {beats}; Hi-ToM cot ≥0.76: {keeps})")


if __name__ == "__main__":
    main()
