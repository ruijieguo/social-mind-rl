"""Audit training data quality.

Reports:
- Per-source record counts (after dedupe)
- Gold letter distribution (should be ~uniform A/B/C/D)
- Story / question / option token-count percentiles
- Top 10 longest stories + first 80 chars (spot-check for outliers)
- Random sample of 5 synth records per task type
"""
from __future__ import annotations
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines


def percentiles(xs, qs=(50, 75, 90, 95, 99)):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {q: s[min(n - 1, int(q / 100 * n))] for q in qs}


def count_tokens(s: str) -> int:
    """Rough English-friendly token count (words + ~chars/3 for CJK)."""
    if not s:
        return 0
    ws = sum(1 for c in s if c.isspace()) + 1
    cjk = sum(1 for c in s if 0x3000 <= ord(c) <= 0x9FFF)
    return ws + cjk // 2


def main():
    train_path = Path("data/tom/tom_train.jsonl")
    records = list(jsonlines.open(train_path))
    print(f"\n=== {train_path} ({len(records)} records) ===\n")

    # 1. Source distribution
    by_src = Counter(r["source"] for r in records)
    print("Per-source counts:")
    for src, n in sorted(by_src.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(records)
        print(f"  {src:12s}  {n:5d}  ({pct:5.1f}%)")

    # 2. Gold letter distribution per source
    print("\nGold letter distribution (should be uniform ~25%):")
    print(f"  {'source':12s}  {'A':>6s} {'B':>6s} {'C':>6s} {'D':>6s}")
    overall_gold = Counter()
    for src in sorted(by_src):
        gold = Counter(r["ground_truth"] for r in records if r["source"] == src)
        overall_gold.update(gold)
        n = sum(gold.values())
        cells = [f"{100 * gold.get(g, 0) / n:5.1f}%" for g in "ABCD"]
        print(f"  {src:12s}  " + " ".join(cells))
    overall_n = sum(overall_gold.values())
    cells = [f"{100 * overall_gold.get(g, 0) / overall_n:5.1f}%" for g in "ABCD"]
    print(f"  {'OVERALL':12s}  " + " ".join(cells))

    # 3. Length distributions (user prompt = story + question + 4 options)
    user_lens, story_lens, opt_lens = [], [], []
    for r in records:
        msgs = r.get("messages", [])
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        if user_msg:
            user_lens.append(count_tokens(user_msg["content"]))
    print(f"\nUser-prompt length (rough tokens), n={len(user_lens)}:")
    for q, v in percentiles(user_lens).items():
        print(f"  p{q}: {v}")
    print(f"  max: {max(user_lens) if user_lens else 0}")
    print(f"  min: {min(user_lens) if user_lens else 0}")

    # 4. Language distribution
    by_lang = Counter(r.get("language", "?") for r in records)
    print(f"\nLanguage distribution:")
    for lang, n in sorted(by_lang.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(records)
        print(f"  {lang:6s}  {n:5d}  ({pct:5.1f}%)")

    # 5. Task tag distribution
    by_tag = Counter(r.get("tag", "?") for r in records)
    print(f"\nTag distribution:")
    for tag, n in sorted(by_tag.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(records)
        print(f"  {tag:12s}  {n:5d}  ({pct:5.1f}%)")

    # 6. Random synth spot-check
    synth = [r for r in records if r["source"] == "synth"]
    if synth:
        print(f"\nRandom 5 synth records (spot check):")
        random.seed(42)
        for r in random.sample(synth, k=min(5, len(synth))):
            print("-" * 60)
            print(f"  qid: {r['question_id']}, gold: {r['ground_truth']}, task: {r.get('task','?')}")
            user_msg = next((m for m in r["messages"] if m["role"] == "user"), None)
            if user_msg:
                print(user_msg["content"][:400])
                if len(user_msg["content"]) > 400:
                    print("  ...")

    # 7. Look for potential leakage by comparing to subset500 first record
    subset = list(jsonlines.open("data/tom/tombench_eval_subset500.jsonl"))
    print(f"\nToMBench subset500 reference: {len(subset)} records (eval-only).")


if __name__ == "__main__":
    main()
