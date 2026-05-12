"""Plot training curves from TensorBoard logs into a single PNG.

Reads output/tensorboard/ and produces output/analysis/curves.png.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


TAGS = [
    ("critic/rewards/mean",                    "Reward (mean)"),
    ("reward/r_fmt_mean",                      "R_fmt"),
    ("reward/r_out_mean",                      "R_out"),
    ("reward/r_len_mean",                      "R_len"),
    ("actor/loss",                             "Actor loss"),
    ("actor/kl",                               "KL"),
    ("actor/entropy",                          "Entropy"),
    ("actor/ppo_ratio_high_clipfrac",          "Clip high frac"),
    ("actor/ppo_ratio_low_clipfrac",           "Clip low frac"),
    ("response_length/mean",                   "Response len (mean)"),
    ("validation/tombench_subset500_accuracy", "Subset500 acc"),
    ("reward/r_total_mean",                    "R_total"),
]


def latest_event_dir(root: Path) -> Path | None:
    events = sorted(root.glob("**/events.out.tfevents.*"),
                    key=lambda p: p.stat().st_mtime)
    return events[-1].parent if events else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb", default="output/tensorboard")
    p.add_argument("--out", default="output/analysis/curves.png")
    args = p.parse_args()

    root = Path(args.tb)
    tb_dir = latest_event_dir(root)
    if tb_dir is None:
        print(f"No tensorboard events under {root}")
        return

    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])

    n = len(TAGS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for i, (tag, title) in enumerate(TAGS):
        ax = axes[i // cols][i % cols]
        if tag in available:
            scalars = ea.Scalars(tag)
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            ax.plot(steps, values)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
