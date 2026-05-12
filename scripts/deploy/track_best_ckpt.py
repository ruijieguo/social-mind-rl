"""Maintain a `best_checkpoint` symlink on TRAIN pointing at the highest
ToMBench subset500 score across all checkpoints.

Runs on TRAIN (inside or alongside the train container). Reads tensorboard
event files for the `validation/tombench_subset500_accuracy` scalar.
"""
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def find_latest_event_dir(tb_root: Path) -> Path | None:
    candidates = sorted(tb_root.glob("**/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1].parent


def get_best_step(tb_dir: Path, tag: str) -> tuple[int, float] | None:
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    best = max(ea.Scalars(tag), key=lambda s: s.value)
    return best.step, best.value


def find_ckpt_for_step(ckpt_root: Path, step: int) -> Path | None:
    cand = list(ckpt_root.glob(f"checkpoint-{step}"))
    if cand:
        return cand[0]
    # ROLL may name checkpoints differently
    cand = list(ckpt_root.glob(f"*step{step}*"))
    return cand[0] if cand else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True,
                   help="dir containing checkpoint-N subdirs")
    p.add_argument("--tb-root", required=True,
                   help="dir containing TensorBoard event files")
    p.add_argument("--tag", default="validation/tombench_subset500_accuracy")
    p.add_argument("--best-symlink", default=None,
                   help="symlink path; default <ckpt-root>/best_checkpoint")
    p.add_argument("--loop", action="store_true", help="re-check every 60s")
    args = p.parse_args()

    ckpt_root = Path(args.ckpt_root)
    tb_root = Path(args.tb_root)
    sym = Path(args.best_symlink) if args.best_symlink else ckpt_root / "best_checkpoint"

    while True:
        tb_dir = find_latest_event_dir(tb_root)
        if tb_dir is None:
            print("waiting for tb events ...")
        else:
            res = get_best_step(tb_dir, args.tag)
            if res is None:
                print(f"tag {args.tag} not yet in tb")
            else:
                step, score = res
                ckpt = find_ckpt_for_step(ckpt_root, step)
                if ckpt is None:
                    print(f"best step {step} ({score:.4f}) but ckpt dir not found")
                else:
                    if sym.exists() or sym.is_symlink():
                        sym.unlink()
                    sym.symlink_to(ckpt.resolve())
                    print(f"best step {step} ({score:.4f}) → {sym} -> {ckpt}")
        if not args.loop:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
