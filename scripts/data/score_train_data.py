"""Offline base-model scorer for curriculum learning (Phase 2 / Stage 10).

For each training record, sample N=8 from a base model (or current best) with
temperature 0.7 and compute the success rate (= fraction of samples that
arrive at the gold answer). This success rate becomes the difficulty score
that drives the easy/medium/hard bucketing for curriculum learning.

Uses vLLM serve endpoint (any healthy model can be the "judge" of difficulty —
we use the current best 14B stage8 by default).

Output:
  data/tom/tom_train_scored.jsonl
  Each record from tom_train.jsonl + {success_rate, difficulty_bucket}
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


_BOXED = re.compile(r"\\boxed\{([A-D])\}")


def score_one(client, record, n_samples=8, model="qwen3-14b-tom-stage8", max_retries=2):
    """Sample N times, return success_rate as fraction matching gold."""
    msgs = record["messages"]
    gold = record["ground_truth"]
    correct = 0
    completed = 0
    for _ in range(n_samples):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=0.7,
                    max_tokens=256,
                    timeout=60,
                )
                content = resp.choices[0].message.content or ""
                m = _BOXED.search(content)
                if m and m.group(1) == gold:
                    correct += 1
                completed += 1
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    completed += 1  # count failure as wrong
    if completed == 0:
        return None
    success_rate = correct / completed
    return success_rate


def bucket(success_rate):
    if success_rate >= 0.75:
        return "easy"
    elif success_rate >= 0.25:
        return "medium"
    else:
        return "hard"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/tom/tom_train.jsonl")
    p.add_argument("--out", default="data/tom/tom_train_scored.jsonl")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="qwen3-14b-tom-stage8")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[score] loaded {len(records)} records", file=sys.stderr)

    done_qids = set()
    if args.resume and Path(args.out).exists():
        with open(args.out) as f:
            for line in f:
                try:
                    done_qids.add(json.loads(line)["question_id"])
                except Exception:
                    continue
        records = [r for r in records if r["question_id"] not in done_qids]
        print(f"[score] resume: {len(done_qids)} done, {len(records)} remaining", file=sys.stderr)

    if args.limit > 0:
        records = records[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    write_lock = threading.Lock()
    counter = {"easy": 0, "medium": 0, "hard": 0, "fail": 0}
    started = time.time()
    with out_path.open(mode, encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(score_one, client, rec, args.n_samples, args.model): rec for rec in records}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            sr = fut.result()
            with write_lock:
                if sr is None:
                    counter["fail"] += 1
                else:
                    bk = bucket(sr)
                    rec_out = {**rec, "success_rate": sr, "difficulty_bucket": bk}
                    fp.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                    fp.flush()
                    counter[bk] += 1
                if i % 50 == 0 or i == len(futures):
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(futures) - i) / rate if rate > 0 else 0
                    print(f"[score] {i}/{len(futures)} easy={counter['easy']} med={counter['medium']} "
                          f"hard={counter['hard']} fail={counter['fail']} rate={rate:.1f}/s eta={eta:.0f}s",
                          file=sys.stderr, flush=True)
    print(f"[score] DONE: easy={counter['easy']} med={counter['medium']} hard={counter['hard']} fail={counter['fail']}")
    print(f"[score] output: {out_path}")


if __name__ == "__main__":
    main()
