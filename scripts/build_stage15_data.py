#!/usr/bin/env python3
"""
Build Stage 15 8B training data:
  1. Filter out records where 8B Stage 7 already gets reward >= 0.95
     (these provide no learning signal under difficulty mask)
  2. Resample remaining records by per-task multiplier from 8B Stage 7's
     own per-task del_tom acc (not 14B's)

Inputs:
  - production_frozen/v3.0/data/tom_train_14b_stage12.jsonl (12519 source records)
  - output/eval/8b_stage7_reward_full12519.jsonl (per-record 8 samples reward, _idx aligned)
  - output/eval/8b_stage7_per_task_acc.json (8B Stage 7 del_tom per-task acc)

Output:
  - data/tom/tom_train_stage15_8b_filtered_weighted.jsonl
"""
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

random.seed(42)

SRC = "production_frozen/v3.0/data/tom_train_14b_stage12.jsonl"
REWARD = "output/eval/8b_stage7_reward_full12519.jsonl"
TASK_ACC = "output/eval/8b_stage7_per_task_acc.json"
OUT = "data/tom/tom_train_stage15_8b_filtered_weighted.jsonl"

REWARD_HIGH = 0.95   # drop sample if mean reward >= this (already mastered)
REWARD_LOW = -0.001  # keep all hard samples (we want them); could set 0.05 to drop pure noise

def multiplier(acc, slope=3.25, offset=0.5, lo=0.7, hi=2.0):
    """Same formula as Stage 14b; produces per-task multiplier."""
    m = 2.0 - (acc - offset) * slope
    return max(lo, min(hi, m))

def main():
    # Load source records
    src = []
    with open(SRC) as f:
        for line in f:
            src.append(json.loads(line))
    print(f"Source records: {len(src)}")

    # Load reward labels (keyed by _idx into src)
    rewards = {}
    with open(REWARD) as f:
        for line in f:
            r = json.loads(line)
            rewards[r["_idx"]] = r
    print(f"Reward labels: {len(rewards)}")

    # Load per-task acc
    task_acc = json.load(open(TASK_ACC))
    print("\n8B Stage 7 per-task acc → multiplier:")
    multipliers = {}
    for t, a in sorted(task_acc.items(), key=lambda x: x[1]):
        multipliers[t] = multiplier(a)
        print(f"  {t:<22}: acc={a:.4f}  mult={multipliers[t]:.2f}")

    # ---- Filter pass ----
    kept = []
    drop_high = 0
    drop_no_label = 0
    for i, rec in enumerate(src):
        rew = rewards.get(i)
        if rew is None:
            drop_no_label += 1
            continue
        if rew["reward_mean"] >= REWARD_HIGH:
            drop_high += 1
            continue
        kept.append((i, rec, rew["reward_mean"]))
    print(f"\nFilter pass:")
    print(f"  drop reward_mean >= {REWARD_HIGH}: {drop_high}")
    print(f"  drop no_label (lost): {drop_no_label}")
    print(f"  kept: {len(kept)}")

    # ---- Reweight pass ----
    out_records = []
    by_task_kept = Counter(r[1].get("task", "Other") for r in kept)
    by_task_out = Counter()

    for idx, rec, rew_mean in kept:
        task = rec.get("task", "Other")
        mult = multipliers.get(task, 1.0)
        # Probabilistic resampling: floor + Bernoulli on remainder
        n = math.floor(mult)
        p = mult - n
        if random.random() < p:
            n += 1
        for _ in range(n):
            out_records.append(rec)
        by_task_out[task] += n

    random.shuffle(out_records)

    print(f"\nFinal: {len(out_records)} records")
    print(f"\nPer-task: kept (filter) → out (reweight)")
    print(f"{'task':<22} {'mult':>5} {'kept':>6} {'out':>6}")
    for t in sorted(by_task_kept):
        m = multipliers.get(t, 1.0)
        print(f"  {t:<22} {m:>5.2f} {by_task_kept[t]:>6} {by_task_out[t]:>6}")

    # Reward distribution of kept records
    kept_rew = [k[2] for k in kept]
    print(f"\nKept reward distribution:")
    bins = [(0.0, 0.0), (0.0, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.80), (0.80, 0.94)]
    for lo, hi in bins:
        n = sum(1 for r in kept_rew if (lo == hi == 0.0 and r == 0.0) or (lo < r <= hi))
        print(f"  {lo:.2f} ~ {hi:.2f}: {n} ({n/len(kept_rew)*100:.1f}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote → {OUT}")

if __name__ == "__main__":
    main()
