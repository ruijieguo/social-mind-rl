"""GPT-5.5 audit of the FULL ToMBench eval set (5718 questions).

This is Phase C.1 of the improvement plan. Goal: identify and exclude
questions where the gold label is wrong, ambiguous, or translation-artifact.

Distinct from gpt55_audit_eval.py (which audited only "both-wrong" cases).
Here we audit every question. Expected ~40% to have label issues based on
the both-wrong sample, but the rate on a representative sample should be
lower (the both-wrong sample is biased toward hard/ambiguous questions).

Output:
  output/analysis/gpt55_eval_full_audit.jsonl
  output/analysis/clean_eval_qids.json    -- set of question_ids to KEEP

A question is KEPT if GPT-5.5's label_assessment is "correct" with
confidence >= "medium". Otherwise excluded.
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


SYSTEM = """You are an expert reviewer of theory-of-mind multiple-choice questions.

You will receive a story, a question, four options A/B/C/D, and the labeled correct answer.

Read the story carefully, then output a JSON object with these fields:

{
  "your_answer": "A" | "B" | "C" | "D",
  "your_reasoning": "<2-3 sentences>",
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

Output the JSON object now."""


def parse_user_prompt_to_fields(user_prompt):
    if user_prompt.startswith("Story:") or user_prompt.startswith("故事："):
        sm = "Story:" if user_prompt.startswith("Story:") else "故事："
        qm = "Question:" if "Question:" in user_prompt else "问题："
    else:
        return None
    _, _, after = user_prompt.partition(sm)
    after = after.lstrip("\n")
    story, _, qopts = after.partition(qm)
    lines = qopts.strip().split("\n")
    q = lines[0].strip()
    opts = {"A": "", "B": "", "C": "", "D": ""}
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("A."): opts["A"] = line[2:].strip()
        elif line.startswith("B."): opts["B"] = line[2:].strip()
        elif line.startswith("C."): opts["C"] = line[2:].strip()
        elif line.startswith("D."): opts["D"] = line[2:].strip()
    return {"story": story.strip(), "question": q, **{f"opt_{k.lower()}": v for k, v in opts.items()}}


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_audit(text):
    if not text: return None
    m = _FENCE.search(text)
    if m: text = m.group(1)
    m = _OBJ.search(text)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def audit_one(client, qid, lang, task, gold, fields, model="gpt-5.5", max_retries=3):
    user = USER_TMPL.format(story=fields["story"], question=fields["question"],
                             opt_a=fields["opt_a"], opt_b=fields["opt_b"],
                             opt_c=fields["opt_c"], opt_d=fields["opt_d"],
                             gold=gold)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.0, max_tokens=400, timeout=60,
            )
            obj = parse_audit(resp.choices[0].message.content or "")
            if obj and "label_assessment" in obj:
                return {"qid": qid, "lang": lang, "task": task, "gold": gold, **obj}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tom/tombench_eval.jsonl")
    p.add_argument("--out", default="output/analysis/gpt55_eval_full_audit.jsonl")
    p.add_argument("--clean-qids", default="output/analysis/clean_eval_qids.json")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="skip qids already in output file")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    items = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            qid = r.get("question_id")
            user = next((m["content"] for m in r.get("messages", []) if m.get("role") == "user"), "")
            fields = parse_user_prompt_to_fields(user)
            if not fields: continue
            items.append({
                "qid": qid, "lang": r.get("language", ""),
                "task": r.get("task", ""), "gold": r.get("gold") or r.get("ground_truth", ""),
                "fields": fields,
            })

    done = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try: done.add(json.loads(line)["qid"])
                except: pass
        items = [it for it in items if it["qid"] not in done]
        print(f"[c1-audit] resume: {len(done)} done, {len(items)} remaining", file=sys.stderr)

    if args.limit > 0:
        items = items[: args.limit]
    print(f"[c1-audit] auditing {len(items)} questions with concurrency={args.concurrency}", file=sys.stderr)

    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(audit_one, client, it["qid"], it["lang"], it["task"], it["gold"], it["fields"], args.model) for it in items]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with write_lock:
                if r:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n"); fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[c1-audit] {i}/{total} ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={(total-i)/rate:.0f}s",
                          file=sys.stderr, flush=True)

    # Build clean_eval_qids: keep questions where label_assessment == correct AND confidence in {high, medium}
    keep = []
    drop = []
    with out_path.open() as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            la = r.get("label_assessment", "wrong")
            lc = r.get("label_confidence", "low")
            if la == "correct" and lc in {"high", "medium"}:
                keep.append(r["qid"])
            else:
                drop.append({"qid": r["qid"], "assessment": la, "confidence": lc, "issue": r.get("issue_category")})
    Path(args.clean_qids).write_text(json.dumps({"keep": keep, "drop_sample": drop[:50], "n_keep": len(keep), "n_drop": len(drop)}, ensure_ascii=False, indent=2))
    print(f"[c1-audit] DONE: {len(keep)} keep, {len(drop)} drop ({len(drop)*100/(len(keep)+len(drop)):.1f}% removal). Clean qids → {args.clean_qids}")


if __name__ == "__main__":
    main()
