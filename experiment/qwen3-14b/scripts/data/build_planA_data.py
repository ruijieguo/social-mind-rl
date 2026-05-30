"""Build the Plan-A (Stage 22) RLVR training data from the v3.1 weighted set.

Two fixes (see experiment/qwen3-14b-full-eval/output/data_audit_2026-05-30.md):
  1. RE-TAG: set tag="tom_mcq" on every record. The v3.1 set left ExploreToM v2 +
     synth_gpt55_phase_d_hot at tag=None, so ROLL routed them to domain "math_rule"
     and silently dropped 26% of the data (the hardest higher-order ToM). Re-tagging
     puts them back in the tom_mcq domain that actually trains.
  2. GOLD REBALANCE: per record, randomly permute the 4 option texts and relabel
     A-D, moving the gold to a (uniform-random) new position. The v3.1 set was
     A=45% / B=31% / C=14% / D=10% — a position prior that matches no eval set.
     A uniform per-record shuffle makes gold ~25% each and decouples position from
     correctness.

Parsing is fail-safe: a record is only reshuffled if its user message ends in a
clean contiguous block of exactly-N labeled option lines (A., B., ...) and the gold
letter is among them; otherwise options are left untouched (record is still re-tagged).

Usage:
  python scripts/data/build_planA_data.py \
    --in data/tom/tom_train_stage14_weighted.jsonl \
    --out data/tom/tom_train_stage22_planA.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

OPT_RE = re.compile(r"^\s*([A-Z])[.．、:：]\s?(.*)$")


def split_options(user: str):
    """Return (head, letters, texts) if the user msg ends in a clean N-option block,
    else None. head is everything before the option block (kept verbatim)."""
    lines = user.split("\n")
    # walk backwards collecting contiguous trailing option lines
    opt_lines = []
    i = len(lines) - 1
    while i >= 0:
        m = OPT_RE.match(lines[i])
        if m and lines[i].strip():
            opt_lines.append((m.group(1), m.group(2)))
            i -= 1
        else:
            break
    opt_lines.reverse()
    if len(opt_lines) < 2:
        return None
    letters = [L for L, _ in opt_lines]
    expected = [chr(ord("A") + j) for j in range(len(opt_lines))]
    if letters != expected:
        return None  # not a clean A,B,C,... block
    head = "\n".join(lines[: len(lines) - len(opt_lines)])
    texts = [t for _, t in opt_lines]
    return head, letters, texts


def rebalance(user: str, gold: str, rng: random.Random):
    """Permute options; return (new_user, new_gold) or None if not parseable."""
    parsed = split_options(user)
    if parsed is None:
        return None
    head, letters, texts = parsed
    if gold not in letters:
        return None
    gold_idx = letters.index(gold)
    perm = list(range(len(texts)))
    rng.shuffle(perm)
    # perm[new_pos] = old_idx
    new_texts = [texts[perm[j]] for j in range(len(texts))]
    new_gold_pos = perm.index(gold_idx)
    new_gold = chr(ord("A") + new_gold_pos)
    opt_block = "\n".join(f"{chr(ord('A') + j)}. {new_texts[j]}" for j in range(len(new_texts)))
    new_user = (head + "\n" + opt_block) if head else opt_block
    return new_user, new_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rebalance", action="store_true", help="only re-tag, skip gold shuffle")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    n = retagged = reshuffled = skipped = 0
    gold_before = collections.Counter()
    gold_after = collections.Counter()
    with open(args.inp) as f, open(args.out, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            gold_before[r["ground_truth"]] += 1

            # fix 1: re-tag
            if r.get("tag") != "tom_mcq":
                r["tag"] = "tom_mcq"
                retagged += 1

            # fix 2: gold rebalance
            if not args.no_rebalance:
                msgs = r["messages"]
                ui = next((k for k, m in enumerate(msgs) if m["role"] == "user"), None)
                if ui is not None:
                    res = rebalance(msgs[ui]["content"], r["ground_truth"], rng)
                    if res is not None:
                        msgs[ui]["content"], r["ground_truth"] = res
                        reshuffled += 1
                    else:
                        skipped += 1
            gold_after[r["ground_truth"]] += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"records:        {n}")
    print(f"re-tagged:      {retagged}  (were tag!=tom_mcq → now all train)")
    print(f"gold reshuffled:{reshuffled}   skipped(unparseable):{skipped}")
    print(f"gold BEFORE:    {dict(sorted(gold_before.items()))}")
    print(f"gold AFTER:     {dict(sorted(gold_after.items()))}")


if __name__ == "__main__":
    main()
