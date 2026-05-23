#!/usr/bin/env python3
"""
Score every record in tom_train_stage12.jsonl (12519) with 8B Stage 7
to identify which samples are too easy (always correct) vs in the
learnable middle (mixed correct/wrong).

Output: a parallel jsonl with reward_mean / reward_n_correct fields,
ready for filtering+reweighting in build_stage15_data.py.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

INPUT = "/data_in/tom_train_stage12.jsonl"  # mounted from tom-data
OUTPUT = "/workspace/output/eval/8b_stage7_reward_full12519.jsonl"
N_SAMPLES = 8
TEMP = 0.99
TOP_P = 0.95
MAX_TOKENS = 256
MODEL = os.getenv("MODEL", "eval-target-8b-stage7")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
CONCURRENCY = int(os.getenv("CONCURRENCY", "32"))

LETTER_RE = re.compile(r'\b([ABCD])\b')

def extract(text):
    m = LETTER_RE.search(text or "")
    return m.group(1) if m else None

def main():
    records = []
    with open(INPUT) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {INPUT}", flush=True)

    # Resume support: skip records already scored
    scored_ids = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            for line in f:
                try:
                    scored_ids.add(json.loads(line).get("_idx"))
                except Exception:
                    pass
        print(f"Resume: {len(scored_ids)} already scored", flush=True)

    todo = [(i, r) for i, r in enumerate(records) if i not in scored_ids]
    print(f"To score: {len(todo)}", flush=True)

    client = OpenAI(api_key="dummy", base_url=BASE_URL, max_retries=2, timeout=120)

    out_f = open(OUTPUT, "a", buffering=1)

    def evaluate_one(idx_rec):
        idx, rec = idx_rec
        msgs = rec["messages"]
        gold = rec["ground_truth"]
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs,
                temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS, n=N_SAMPLES,
            )
            preds = [extract(c.message.content) for c in resp.choices]
            scores = [1 if p == gold else 0 for p in preds]
        except Exception as e:
            return None, str(e)
        return {
            "_idx": idx,
            "task": rec.get("task"),
            "language": rec.get("language"),
            "n_correct": sum(scores),
            "n_total": N_SAMPLES,
            "reward_mean": sum(scores) / N_SAMPLES,
            "gold": gold,
        }, None

    t0 = time.time()
    n_done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(evaluate_one, ir): ir[0] for ir in todo}
        for fut in as_completed(futures):
            res, err = fut.result()
            if res is not None:
                out_f.write(json.dumps(res) + "\n")
                n_done += 1
            else:
                n_err += 1
            if (n_done + n_err) % 200 == 0:
                elapsed = time.time() - t0
                rate = (n_done + n_err) / elapsed if elapsed > 0 else 0
                eta = (len(todo) - (n_done + n_err)) / rate if rate > 0 else 0
                print(f"  done={n_done} err={n_err} elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.0f}s", flush=True)
    out_f.close()
    print(f"\nDone: {n_done} ok, {n_err} errors, {time.time()-t0:.0f}s total", flush=True)

if __name__ == "__main__":
    main()
