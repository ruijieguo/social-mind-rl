"""Quality-filter synth records via deepseek-v4-flash LLM-as-judge.

For each synth record, ask the model:
  1. Is this genuinely a theory-of-mind question?
  2. Does it have exactly one correct answer (the gold letter)?
  3. Are the other three options plausible but wrong?

Records that fail any check are flagged. We keep the training file intact
but emit a filter-report with the list of bad qids + reasons, so the user
can decide the retention threshold.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
from openai import OpenAI


JUDGE_SYSTEM = (
    "You are an expert evaluator of theory-of-mind multiple-choice training data. "
    "Given a story, question, 4 options and the claimed gold answer letter, output a JSON "
    "with keys: is_tom (bool), unique_gold (bool), distractors_plausible (bool), overall_good (bool), "
    "reason (short string). Only one of the four options should be correct; the claimed gold must "
    "match your judgment; distractors should be plausible but wrong. Be strict."
)


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_judge_response(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    m = _OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    for key in ("is_tom", "unique_gold", "distractors_plausible", "overall_good"):
        if key not in obj or not isinstance(obj[key], bool):
            return None
    return obj


def judge_once(client: OpenAI, record_text: str, gold: str, model: str, max_retries: int = 3) -> Optional[dict]:
    user = f"{record_text}\n\nClaimed gold answer: {gold}\n\nEmit the JSON judgment."
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=300,
                timeout=60,
            )
            parsed = parse_judge_response(resp.choices[0].message.content or "")
            if parsed:
                return parsed
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/tom/tom_train.jsonl")
    p.add_argument("--only-source-tag", default="synth",
                   help="only judge records whose source starts with this (synth, synth_zh, ...)")
    p.add_argument("--n", type=int, default=500,
                   help="how many records to judge (random sample)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--out", default="data/tom/synth_quality_report.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    records = list(jsonlines.open(args.source))
    candidates = [r for r in records if r.get("source", "").startswith(args.only_source_tag)]
    rng = random.Random(args.seed)
    sample = rng.sample(candidates, k=min(args.n, len(candidates)))
    print(f"[judge] source={args.source}  candidates={len(candidates)}  sample={len(sample)}",
          file=sys.stderr, flush=True)

    write_lock = threading.Lock()
    results = []
    counter = {"ok": 0, "bad": 0, "failed": 0}
    started = time.time()

    def task(rec):
        user_msg = next((m for m in rec["messages"] if m["role"] == "user"), None)
        if not user_msg:
            return rec, None
        return rec, judge_once(client, user_msg["content"], rec["ground_truth"], args.model)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(task, r) for r in sample]
        total = len(futures)
        for i, f in enumerate(as_completed(futures), 1):
            rec, verdict = f.result()
            with write_lock:
                if verdict is None:
                    counter["failed"] += 1
                elif verdict.get("overall_good"):
                    counter["ok"] += 1
                else:
                    counter["bad"] += 1
                    results.append({
                        "question_id": rec.get("question_id"),
                        "source": rec.get("source"),
                        "task": rec.get("task"),
                        "gold": rec.get("ground_truth"),
                        "verdict": verdict,
                    })
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    print(f"[judge] {i}/{total}  ok={counter['ok']}  bad={counter['bad']}  failed={counter['failed']}  "
                          f"elapsed={elapsed:.0f}s", file=sys.stderr, flush=True)

    n = counter["ok"] + counter["bad"]
    good_rate = counter["ok"] / n if n else 0.0
    print(f"\n[judge] FINAL: n={n}  ok={counter['ok']} ({100*good_rate:.1f}%)  bad={counter['bad']}  failed={counter['failed']}",
          file=sys.stderr, flush=True)

    out = Path(args.out)
    out.write_text(json.dumps({
        "model": args.model,
        "sample_size": len(sample),
        "ok": counter["ok"],
        "bad": counter["bad"],
        "failed": counter["failed"],
        "good_rate": good_rate,
        "bad_records": results,
    }, ensure_ascii=False, indent=2))
    print(f"[judge] wrote bad-record list to {out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
