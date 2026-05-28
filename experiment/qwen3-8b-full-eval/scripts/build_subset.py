"""Build a deterministic 200-question subset of ToMBench / Hi-ToM for DashScope
del_tom evaluation, where the full set × 8 samples is too slow for the API.

Stratified sample over (task, language) for ToMBench, and over `task` (=order_*)
for Hi-ToM, with seed=42.

Outputs:
  data/subsets/tombench_eval_subset200_seed42.jsonl
  data/subsets/hitom_eval_subset200_seed42.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p: Path, recs: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(records: list[dict], strat_key, n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_strat: dict = defaultdict(list)
    for r in records:
        by_strat[strat_key(r)].append(r)
    keys = sorted(by_strat.keys())

    # Allocate quota proportionally, with at least 1 per stratum
    total = len(records)
    quotas = {}
    for k in keys:
        quotas[k] = max(1, int(round(n * len(by_strat[k]) / total)))
    # Adjust to hit exactly n
    while sum(quotas.values()) > n:
        # Trim from the largest stratum
        biggest = max(keys, key=lambda k: quotas[k] / max(1, len(by_strat[k])))
        if quotas[biggest] > 1:
            quotas[biggest] -= 1
        else:
            break
    while sum(quotas.values()) < n:
        smallest = min(keys, key=lambda k: quotas[k])
        if quotas[smallest] < len(by_strat[smallest]):
            quotas[smallest] += 1
        else:
            break

    sampled = []
    for k in keys:
        pool = by_strat[k]
        rng.shuffle(pool)
        sampled.extend(pool[: quotas[k]])
    rng.shuffle(sampled)
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tombench-in", required=True)
    ap.add_argument("--hitom-in", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    tom = load_jsonl(Path(args.tombench_in))
    hit = load_jsonl(Path(args.hitom_in))

    tom_sub = stratified_sample(tom, lambda r: (r["task"], r["language"]), args.n, args.seed)
    hit_sub = stratified_sample(hit, lambda r: r["task"], args.n, args.seed)

    write_jsonl(out_dir / f"tombench_eval_subset{args.n}_seed{args.seed}.jsonl", tom_sub)
    write_jsonl(out_dir / f"hitom_eval_subset{args.n}_seed{args.seed}.jsonl", hit_sub)

    print(f"ToMBench subset: {len(tom_sub)} from {len(tom)} (strat by task,lang)")
    from collections import Counter
    print("  ", Counter((r["task"], r["language"]) for r in tom_sub))
    print(f"Hi-ToM subset: {len(hit_sub)} from {len(hit)} (strat by order)")
    print("  ", Counter(r["task"] for r in hit_sub))


if __name__ == "__main__":
    main()
