"""GPT-5.5 audit: for a sample of "both wrong" cases (where 14B-tom AND
deepseek-v4-pro both got it wrong), ask GPT-5.5 to:

  (1) Solve the question itself, blind (no reference to gold)
  (2) Then judge whether the gold label is correct, ambiguous, or wrong

If GPT-5.5 confidently disagrees with the gold AND agrees with the model
prediction, that is evidence of a label issue.

Output: output/analysis/gpt55_audit.jsonl (one record per audited question)
        output/analysis/gpt55_audit_summary.md (aggregated)
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


SYSTEM = """You are an expert reviewer of theory-of-mind multiple-choice questions.

You will receive a story, a question, four options A/B/C/D, the labeled correct answer, and (optionally) the answers two AI systems gave that disagree with the label.

Your job: Read the story carefully, then output a JSON object with these fields:

{
  "your_answer": "A" | "B" | "C" | "D",
  "your_reasoning": "<2-3 sentences explaining your choice>",
  "label_assessment": "correct" | "ambiguous" | "wrong",
  "label_confidence": "high" | "medium" | "low",
  "issue_category": "label_correct" | "ambiguous_question" | "wrong_label" | "translation_artifact" | "options_overlap"
}

Be honest. If the gold label is wrong, say so. If the question is ambiguous and multiple answers are defensible, say so. Do NOT default to label_correct out of deference to the dataset.
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

For context:
- Model qwen3-14b-tom answered: {our_pred}
- Model deepseek-v4-pro answered: {ds_pred}
- (Both these models got it "wrong" by the gold label, so we are auditing whether the gold label itself is reliable.)

Output the JSON object now."""


def parse_user_prompt_to_fields(user_prompt):
    if user_prompt.startswith("Story:") or user_prompt.startswith("故事："):
        story_marker = "Story:" if user_prompt.startswith("Story:") else "故事："
        question_marker = "Question:" if "Question:" in user_prompt else "问题："
    else:
        return None
    story_part, _, after_story = user_prompt.partition(story_marker)
    after_story = after_story.lstrip("\n")
    story, _, after_q_marker = after_story.partition(question_marker)
    story = story.strip()
    question_and_options = after_q_marker.strip()
    lines = question_and_options.split("\n")
    question_line = lines[0].strip()
    opts = {"A": "", "B": "", "C": "", "D": ""}
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("A."): opts["A"] = line[2:].strip()
        elif line.startswith("B."): opts["B"] = line[2:].strip()
        elif line.startswith("C."): opts["C"] = line[2:].strip()
        elif line.startswith("D."): opts["D"] = line[2:].strip()
    return {"story": story, "question": question_line, **{f"opt_{k.lower()}": v for k, v in opts.items()}}


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


def audit_one(client, error_record, model="gpt-5.5", max_retries=3):
    fields = parse_user_prompt_to_fields(error_record["user_prompt"])
    if not fields:
        return None
    user = USER_TMPL.format(
        story=fields["story"],
        question=fields["question"],
        opt_a=fields["opt_a"], opt_b=fields["opt_b"],
        opt_c=fields["opt_c"], opt_d=fields["opt_d"],
        gold=error_record["gold"],
        our_pred=error_record["our_pred"],
        ds_pred=error_record.get("ds_pred", "?"),
    )
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.0, max_tokens=600, timeout=60,
            )
            content = resp.choices[0].message.content or ""
            obj = parse_audit(content)
            if not obj:
                last_err = f"parse fail: {content[:100]!r}"
                continue
            return {
                "qid": error_record["qid"],
                "task": error_record["task"],
                "lang": error_record["lang"],
                "gold": error_record["gold"],
                "our_pred": error_record["our_pred"],
                "ds_pred": error_record.get("ds_pred"),
                "audit": obj,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    print(f"AUDIT FAIL {error_record['qid']}: {last_err}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--errors", default="output/analysis/14b_errors_categorized.json")
    p.add_argument("--out", default="output/analysis/gpt55_audit.jsonl")
    p.add_argument("--n", type=int, default=200, help="how many to audit")
    p.add_argument("--filter", default="both_wrong", choices=["both_wrong", "only_we_wrong", "all"])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    errors = json.loads(Path(args.errors).read_text())
    if args.filter == "both_wrong":
        errors = [e for e in errors if not e.get("ds_correct")]
    elif args.filter == "only_we_wrong":
        errors = [e for e in errors if e.get("ds_correct")]
    print(f"[audit] {args.filter}: {len(errors)} candidates")

    by_task = {}
    for e in errors:
        by_task.setdefault(e["task"], []).append(e)
    rng = random.Random(args.seed)
    per_task = max(1, args.n // len(by_task))
    sample = []
    for t, es in sorted(by_task.items()):
        rng.shuffle(es)
        sample.extend(es[:per_task])
    rng.shuffle(sample)
    sample = sample[:args.n]
    print(f"[audit] sampled {len(sample)} questions across {len(by_task)} tasks")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(audit_one, client, e, args.model): e for e in sample}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            with write_lock:
                if r:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n"); fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 10 == 0 or i == len(futures):
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[audit] {i}/{len(futures)} done ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s elapsed={elapsed:.0f}s",
                          file=sys.stderr, flush=True)
    print(f"[audit] DONE: {counter['ok']} successful audits in {out_path}")


if __name__ == "__main__":
    main()
