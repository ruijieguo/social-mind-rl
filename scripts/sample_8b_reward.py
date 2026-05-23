#!/usr/bin/env python3
"""
Sample 8B Stage 7 model rollouts on stage14b_weighted training data
to evaluate reward distribution. Mimics ROLL's rollout protocol:
- For each prompt, generate 8 samples with temp=0.99, top_p=0.95
- Score each sample (correct = 1.0, wrong = 0.0)
- Compute group mean (the score that gets compared to difficulty thresholds)

Output: distribution of group-mean scores. Decide mask thresholds.
"""
import json
import sys
import os
import asyncio
import random
import re
from collections import defaultdict, Counter
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eval'))
from openai import OpenAI

random.seed(42)

DATA_PATH = "/data_in/tom_train_stage14b_weighted.jsonl"
N_PROMPTS = int(os.getenv("N_PROMPTS", "200"))   # sample size
N_SAMPLES = 8                                    # match num_return_sequences_in_group
TEMPERATURE = 0.99
TOP_P = 0.95
MAX_TOKENS = 256
MODEL = os.getenv("MODEL", "eval-8b-stage7")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
CONCURRENCY = int(os.getenv("CONCURRENCY", "32"))

LETTER_RE = re.compile(r'\b([ABCD])\b')

def extract_letter(text: str) -> str | None:
    m = LETTER_RE.search(text or "")
    return m.group(1) if m else None

def main():
    # Load + sample records
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    # Stratify sample by task to ensure coverage
    by_task = defaultdict(list)
    for r in records:
        by_task[r.get("task", "?")].append(r)
    sampled = []
    per_task_n = max(N_PROMPTS // len(by_task), 5)
    for task, recs in by_task.items():
        random.shuffle(recs)
        sampled.extend(recs[:per_task_n])
    random.shuffle(sampled)
    print(f"Sampling {len(sampled)} prompts across {len(by_task)} tasks ({per_task_n}/task)")

    client = OpenAI(api_key="dummy", base_url=BASE_URL, max_retries=2, timeout=120)

    # Per-prompt: 8 generations, score each, take group mean
    from concurrent.futures import ThreadPoolExecutor

    def evaluate_one(rec):
        msgs = rec["messages"]
        gold = rec["ground_truth"]
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                n=N_SAMPLES,
            )
            preds = [extract_letter(c.message.content) for c in resp.choices]
            scores = [1.0 if p == gold else 0.0 for p in preds]
        except Exception as e:
            print(f"  err: {e}")
            return None
        return {
            "task": rec.get("task"),
            "scores": scores,
            "group_mean": sum(scores) / len(scores),
            "gold": gold,
        }

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for i, r in enumerate(ex.map(evaluate_one, sampled)):
            if r is not None:
                results.append(r)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(sampled)} done")
    print(f"  Total: {len(results)}, took {time.time()-t0:.1f}s")

    # Compute distribution of group means
    means = [r["group_mean"] for r in results]
    bins = [(-0.001, 0.0), (0.0, 0.10), (0.10, 0.15), (0.15, 0.30), (0.30, 0.50),
            (0.50, 0.70), (0.70, 0.80), (0.80, 0.95), (0.95, 0.999), (0.999, 1.001)]
    print("\n=== Group-mean score distribution ===")
    print(f"{'range':<14} {'count':>6} {'%':>7}  cumulative")
    cum = 0
    for lo, hi in bins:
        n = sum(1 for m in means if lo < m <= hi)
        cum += n
        print(f"({lo:>5.3f}, {hi:>5.3f}] {n:>6} {n/len(means)*100:>6.1f}%  {cum/len(means)*100:>6.1f}%")

    # Mask coverage analysis: simulate Stage 12 (0.1, 0.95), Stage 14b (0.15, 0.80), and tighter
    print("\n=== Mask coverage (samples that pass: low < mean < high) ===")
    for low, high in [(0.0, 1.0), (0.1, 0.95), (0.15, 0.95), (0.15, 0.80),
                      (0.10, 0.70), (0.20, 0.70), (0.10, 0.60)]:
        passes = sum(1 for m in means if low < m < high)
        print(f"  ({low:.2f}, {high:.2f}): {passes}/{len(means)} = {passes/len(means)*100:.1f}%")

    # Per-task break
    print("\n=== Per-task group_mean ===")
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r["group_mean"])
    for task, vals in sorted(by_task.items(), key=lambda x: sum(x[1])/len(x[1])):
        avg = sum(vals)/len(vals)
        # Mask coverage at (0.15, 0.80)
        passes = sum(1 for v in vals if 0.15 < v < 0.80)
        print(f"  {task:<22} mean={avg:.3f}  n={len(vals)}  pass(0.15,0.80)={passes}/{len(vals)} ({passes/len(vals)*100:.0f}%)")

    # Save raw
    with open("/workspace/output/eval/8b_stage7_reward_sample.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → /workspace/output/eval/8b_stage7_reward_sample.json")

if __name__ == "__main__":
    main()
