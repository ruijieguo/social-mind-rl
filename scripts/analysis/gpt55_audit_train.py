"""GPT-5.5 audit of training data: sample N records per source and ask GPT-5.5
to check whether (story, question, options, gold) form a valid ToM training
sample.

Why: stage 5's Phase-1 synthesis showed limited transfer to ToMBench.
One hypothesis is that synth data, while well-formatted, has subtle
issues (gold label not actually best, options not exhaustive, story
under-constrains the answer). GPT-5.5 should catch these.

Output: output/analysis/gpt55_train_audit.jsonl
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

from openai import OpenAI


SYSTEM = """You are an expert reviewer of theory-of-mind multiple-choice training data.

You will receive a story, a question, four options A/B/C/D, and the labeled correct answer (gold).

Judge whether this training sample is high-quality. Output a JSON object:

{
  "your_answer": "A" | "B" | "C" | "D",
  "your_reasoning": "<one sentence>",
  "label_correct": true | false,
  "label_confidence": "high" | "medium" | "low",
  "issues": [<zero or more of: "ambiguous_question", "wrong_label", "options_overlap", "story_underconstrains_answer", "non_tom_question", "translation_artifact", "factually_inconsistent">],
  "training_value": "high" | "medium" | "low" | "harmful"
}

`training_value`: how useful is this record for training a Theory-of-Mind model?
- "high": clear, well-constructed, label is unambiguously right
- "medium": defensible but has minor issues
- "low": question is too easy, ambiguous, or off-topic — could mildly confuse training
- "harmful": question is broken or label is clearly wrong — using it would actively hurt the model

Be strict. The goal is to find data problems we can fix.
"""


USER_TMPL = """Story:
{story}

Question: {question}

Options:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Labeled correct answer (gold): {gold}

Output the JSON object now."""


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_audit(text):
    if not text: return None
    m = _FENCE.search(text)
    if m: text = m.group(1)
    m = _OBJ.search(text)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_train_record(line):
    r = json.loads(line)
    msgs = r.get("messages", [])
    user_content = ""
    for m in msgs:
        if m.get("role") == "user":
            user_content = m.get("content", "")
            break
    if not user_content:
        return None
    if user_content.startswith("Story:") or user_content.startswith("故事："):
        story_marker = "Story:" if user_content.startswith("Story:") else "故事："
        question_marker = "Question:" if "Question:" in user_content else "问题："
    else:
        return None
    _, _, after_story = user_content.partition(story_marker)
    story, _, after_q_marker = after_story.lstrip("\n").partition(question_marker)
    story = story.strip()
    lines = after_q_marker.strip().split("\n")
    question_line = lines[0].strip()
    opts = {"A": "", "B": "", "C": "", "D": ""}
    for line2 in lines[1:]:
        line2 = line2.strip()
        if line2.startswith("A."): opts["A"] = line2[2:].strip()
        elif line2.startswith("B."): opts["B"] = line2[2:].strip()
        elif line2.startswith("C."): opts["C"] = line2[2:].strip()
        elif line2.startswith("D."): opts["D"] = line2[2:].strip()
    return {
        "qid": r.get("question_id"),
        "source": r.get("source"),
        "task": r.get("task"),
        "lang": r.get("language"),
        "gold": r.get("ground_truth"),
        "story": story,
        "question": question_line,
        **{f"opt_{k.lower()}": v for k, v in opts.items()},
    }


def audit_one(client, record, model="gpt-5.5", max_retries=3):
    user = USER_TMPL.format(
        story=record["story"], question=record["question"],
        opt_a=record["opt_a"], opt_b=record["opt_b"],
        opt_c=record["opt_c"], opt_d=record["opt_d"],
        gold=record["gold"],
    )
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.0, max_tokens=400, timeout=60,
            )
            content = resp.choices[0].message.content or ""
            obj = parse_audit(content)
            if not obj:
                last_err = f"parse fail: {content[:100]!r}"
                continue
            return {
                "qid": record["qid"], "source": record["source"], "task": record["task"], "lang": record["lang"],
                "gold": record["gold"],
                "audit": obj,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    print(f"AUDIT FAIL {record.get('qid')}: {last_err}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/tom/tom_train.jsonl")
    p.add_argument("--out", default="output/analysis/gpt55_train_audit.jsonl")
    p.add_argument("--n-per-source", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    by_source = {}
    with open(args.train) as f:
        for line in f:
            r = parse_train_record(line)
            if r:
                by_source.setdefault(r["source"], []).append(r)
    print(f"[train-audit] sources: {[(k, len(v)) for k, v in by_source.items()]}", file=sys.stderr)

    rng = random.Random(args.seed)
    sample = []
    for src, recs in by_source.items():
        rng.shuffle(recs)
        sample.extend(recs[:args.n_per_source])
    rng.shuffle(sample)
    print(f"[train-audit] sampled {len(sample)} records ({args.n_per_source}/source)", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(audit_one, client, r, args.model): r for r in sample}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with write_lock:
                if r:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n"); fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 10 == 0 or i == len(futures):
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[train-audit] {i}/{len(futures)} ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s",
                          file=sys.stderr, flush=True)
    print(f"[train-audit] DONE: {counter['ok']} audits in {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
