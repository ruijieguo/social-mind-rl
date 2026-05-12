"""Synthesize ToM MCQ questions via deepseek-v4-pro API.

Generates training-grade ToM MCQ records covering the 9 ToMBench task types.
Each call asks for a fresh question + 4 options + gold letter.
Explicitly prohibits reproducing ToMBench questions.

Writes streamingly to the output file (line-by-line) so a crash
doesn't lose progress, and prints periodic progress to stderr.
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

from openai import OpenAI

from scripts.data.schema import TomRecord


SYNTH_SYSTEM = (
    "You are a careful question writer creating new theory-of-mind multiple-choice questions for training. "
    "Your output MUST be a single JSON object with keys: story, question, options (an object with A,B,C,D), answer (one of A,B,C,D). "
    "Do NOT reproduce, paraphrase, or translate any question from ToMBench by Chen et al. (ACL 2024). "
    "Write entirely new scenarios."
)


SYNTH_TASK_PROMPTS = {
    "False Belief":       "Write a False Belief task: a character's belief differs from reality after an unseen change.",
    "Strange Story":      "Write a Strange Story task involving subtle social misunderstanding or irony.",
    "Unexpected Outcome": "Write an Unexpected Outcome task where the result of an action differs from the character's expectation.",
    "Persuasion Story":   "Write a Persuasion Story task where one character tries to change another's belief.",
    "Knowledge":          "Write a Knowledge-Attention Link task where a character's knowledge depends on what they observed.",
    "Desire":             "Write a Multiple Desires task where two characters have different preferences.",
    "Emotion":            "Write a Discrepant Emotions task where two characters feel differently about the same event.",
    "Intention":          "Write a Prediction of Actions task asking what a character will do given their intention.",
    "Non-literal Comm":   "Write a Hinting Task: a character makes an indirect request and we must infer their actual desire.",
}

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_synth_response(text: str) -> Optional[TomRecord]:
    """Parse model output JSON into TomRecord, returning None on any failure."""
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
    try:
        story = obj["story"]
        question = obj["question"]
        opts = obj["options"]
        answer = obj["answer"]
    except (KeyError, TypeError):
        return None
    if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
        return None
    answer = str(answer).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return None
    return TomRecord(
        question_id="synth_pending",
        source="synth", language="en", task="Other",
        story=str(story), question=str(question),
        opt_a=str(opts["A"]), opt_b=str(opts["B"]),
        opt_c=str(opts["C"]), opt_d=str(opts["D"]),
        gold=answer,
    )


def call_deepseek_once(client: OpenAI, task: str, model: str = "deepseek-v4-flash", max_retries: int = 3) -> Optional[TomRecord]:
    user = SYNTH_TASK_PROMPTS[task] + " Output the JSON object directly."
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.9,
                max_tokens=800,
                timeout=120,
            )
            content = resp.choices[0].message.content or ""
            rec = parse_synth_response(content)
            if rec is None:
                last_err = f"parse failed; content[:100]={content[:100]!r}"
                continue
            rec.task = task
            return rec
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    print(f"[synth] task={task} failed after {max_retries} retries: {last_err}", file=sys.stderr, flush=True)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--task", default="all",
                   help="comma-separated subset of task types, or 'all'")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="data/tom/raw/synth.jsonl")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--model", default="deepseek-v4-flash",
                   help="deepseek model id; flash is ~2x faster than pro for this task")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    if args.task == "all":
        tasks = list(SYNTH_TASK_PROMPTS.keys())
    else:
        tasks = [t.strip() for t in args.task.split(",")]
        for t in tasks:
            if t not in SYNTH_TASK_PROMPTS:
                raise SystemExit(f"unknown task: {t}")
    per_task = max(1, args.n // len(tasks))
    plan = []
    for t in tasks:
        plan.extend([t] * per_task)
    random.shuffle(plan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[synth] starting: n={args.n}, concurrency={args.concurrency}, tasks={len(tasks)}, planned={len(plan)}",
        file=sys.stderr, flush=True,
    )

    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()

    with out.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(call_deepseek_once, client, t, args.model, args.max_retries): t for t in plan}
        total = len(futures)
        for i, f in enumerate(as_completed(futures), 1):
            rec = f.result()
            with write_lock:
                if rec is not None:
                    rec.question_id = f"synth_{counter['ok']}"
                    fp.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(
                        f"[synth] {i}/{total} done | ok={counter['ok']} fail={counter['fail']} | "
                        f"{rate:.1f} req/s | elapsed={elapsed:.0f}s",
                        file=sys.stderr, flush=True,
                    )

    print(
        f"[synth] FINAL: wrote {counter['ok']} synthetic records to {out} (failed: {counter['fail']})",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
