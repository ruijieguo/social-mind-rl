"""
Helper: pick the best Track D / Track E checkpoint based on val trajectory.
Stage 8 had its sweet spot at step ~200 (val 0.706). The final step (350) was
not necessarily best — common in RLVR with dynamic sampling.

Reads the training log, prints val by step, and tells which step had highest val.
"""
import argparse
import json
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_file", help="Path to training log file")
    args = ap.parse_args()

    log = open(args.log_file).read()
    rows = []
    for m in re.finditer(r'metrics_tag: ({.*?"system/step":\s*\d+[^}]*})', log):
        try:
            d = json.loads(m.group(1))
            step = d.get('step')
            metrics = d.get('metrics', {})
            v = metrics.get('val_correct/all/mean')
            if v is not None and isinstance(v, (int, float)):
                rows.append((step, v))
        except Exception:
            pass

    if not rows:
        print("No val measurements found")
        sys.exit(1)

    rows.sort()
    print(f"{'step':<6} {'val':<10}")
    for step, val in rows:
        print(f"{step:<6} {val:.4f}")

    best_step, best_val = max(rows, key=lambda x: x[1])
    print(f"\nBest: step {best_step} with val {best_val:.4f}")
    print(f"Final: step {rows[-1][0]} with val {rows[-1][1]:.4f}")
    if best_step != rows[-1][0]:
        delta = (best_val - rows[-1][1]) * 100
        print(f"Best step was NOT final ({best_step} vs final {rows[-1][0]}, +{delta:.2f}pp)")
        print(f"For convert/eval, use the checkpoint at step {best_step}")


if __name__ == "__main__":
    main()
