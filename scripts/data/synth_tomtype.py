"""Synthesize ToM MCQ questions via deepseek-v4-pro API.

Generates ~1.5k records covering the 8 ToMBench task types.
Each call asks for a fresh question + 4 options + gold letter.
Explicitly prohibits reproducing ToMBench questions.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
from openai import OpenAI
from tqdm import tqdm

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
    # Strip markdown fences
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    # Find outermost JSON object
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


def call_deepseek_once(client: OpenAI, task: str) -> Optional[TomRecord]:
    user = SYNTH_TASK_PROMPTS[task] + " Output the JSON object directly."
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.9,
                max_tokens=800,
                timeout=60,
            )
            rec = parse_synth_response(resp.choices[0].message.content or "")
            if rec is None:
                continue
            rec.task = task
            return rec
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"synth call failed: {e}")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--task", default="all",
                   help="comma-separated subset of task types, or 'all'")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="data/tom/raw/synth.jsonl")
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
    records: list[TomRecord] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_deepseek_once, client, t) for t in plan]
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="synth")):
            rec = f.result()
            if rec is not None:
                rec.question_id = f"synth_{i}"
                records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} synthetic records to {out}")


if __name__ == "__main__":
    main()
