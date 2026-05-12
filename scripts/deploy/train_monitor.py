"""External early-stop monitor for ROLL training.

Watches the latest tensorboard event file. If any termination condition is
met, the monitor calls `docker kill <container>` to terminate training.
Best-checkpoint symlinking continues regardless.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import time
from collections import deque
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def latest_tb_dir(tb_root: Path) -> Path | None:
    events = sorted(tb_root.glob("**/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    return events[-1].parent if events else None


def read_scalar(tb_dir: Path, tag: str):
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def should_stop(tb_dir: Path) -> tuple[bool, str]:
    """Return (should_stop, reason)."""
    # Rule 1: KL > 0.5 for 3 consecutive eval windows
    kl = read_scalar(tb_dir, "actor/kl")
    if len(kl) >= 3 and all(v > 0.5 for _, v in kl[-3:]):
        return True, f"kl>0.5 for 3 consecutive ({[v for _,v in kl[-3:]]})"

    # Rule 2: entropy < 0.1 for 3 consecutive
    ent = read_scalar(tb_dir, "actor/entropy")
    if len(ent) >= 3 and all(v < 0.1 for _, v in ent[-3:]):
        return True, f"entropy<0.1 for 3 consecutive ({[v for _,v in ent[-3:]]})"

    # Rule 3: subset500 acc continuously > 3pp below best
    acc = read_scalar(tb_dir, "validation/tombench_subset500_accuracy")
    if len(acc) >= 3:
        best = max(v for _, v in acc)
        recent = [v for _, v in acc[-3:]]
        if all(v < best - 0.03 for v in recent):
            return True, f"subset500 3 consecutive evals < best-0.03 (best={best:.4f}, recent={recent})"

    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb-root", required=True)
    p.add_argument("--container", required=True,
                   help="docker container name to kill on stop")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()

    tb_root = Path(args.tb_root)
    while True:
        tb_dir = latest_tb_dir(tb_root)
        if tb_dir is not None:
            stop, reason = should_stop(tb_dir)
            if stop:
                print(f"[monitor] EARLY STOP: {reason}")
                subprocess.run(["docker", "kill", args.container], check=False)
                return
            else:
                print(f"[monitor] OK at {time.strftime('%H:%M:%S')}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
