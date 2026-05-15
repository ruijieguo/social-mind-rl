"""Plot training curves from a stdout training log into a single PNG.

Stage1 was run with `track_with: stdout` (no tensorboard files), so the
sibling `plot_training_curves.py` would find nothing. This parses the
`metrics_tag: {...}` JSON lines emitted by ROLL's tracker and plots the
same set of curves.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TAGS = [
    ("critic/score/mean",                "Train score / acc"),
    ("critic/rewards/mean",              "Reward (mean)"),
    ("tom_mcq/reward/r_fmt_mean",        "R_fmt"),
    ("tom_mcq/reward/r_out_mean",        "R_out"),
    ("tom_mcq/reward/r_len_mean",        "R_len"),
    ("tom_mcq/reward/r_total_mean",      "R_total"),
    ("actor/total_loss",                 "Actor loss"),
    ("actor/kl_loss",                    "KL"),
    ("critic/entropy/mean",              "Entropy"),
    ("token/response_length/mean",       "Response len (mean)"),
    ("val_correct/all/mean",             "Subset500 acc"),
    ("actor_train/grad_norm",            "Grad norm"),
]


def parse(path: Path):
    pattern = re.compile(r"metrics_tag: (\{.*\})\s*$")
    series: dict[str, list[tuple[int, float]]] = {tag: [] for tag, _ in TAGS}
    with path.open() as fh:
        for line in fh:
            m = pattern.search(line)
            if not m:
                continue
            try:
                payload = json.loads(m.group(1))
            except Exception:
                continue
            step = payload.get("step")
            if step is None:
                continue
            metrics = payload.get("metrics", {})
            for tag, _ in TAGS:
                v = metrics.get(tag)
                if v is None:
                    continue
                series[tag].append((step, float(v)))
    return series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="stdout training log file")
    p.add_argument("--out", default="output/analysis/curves.png")
    args = p.parse_args()

    series = parse(Path(args.log))

    cols = 3
    rows = (len(TAGS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for i, (tag, title) in enumerate(TAGS):
        ax = axes[i // cols][i % cols]
        pts = series.get(tag, [])
        if pts:
            steps, values = zip(*pts)
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
