"""GPT-5.5 audit of generated reasoning traces.

For each trace:
1. Strip the gold answer from prompt
2. Give GPT-5.5 only story + question + options + reasoning
3. Ask GPT-5.5: is the reasoning sound? does it correctly arrive at an answer?
4. Output: {valid: bool, predicted_answer, issue: <category>}

Issue categories:
  - "sound": reasoning is correct, well-structured, arrives at gold answer
  - "wrong_conclusion": reasoning is OK but concludes wrong answer
  - "circular": just restates the gold without inference
  - "ambiguous": multiple steps don't add up
  - "shortcut": skips key inference step
  - "style_bad": filler / unclear / restates story
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


SYSTEM = """You are an expert reviewer of theory-of-mind reasoning traces written for student model SFT training.

You will see a story, a question, four options, and a reasoning trace (with steps and a boxed answer). You must judge whether the reasoning is sound, well-structured, and arrives at the correct conclusion.

Output ONLY a JSON object:
{
  "verdict": "sound" | "wrong_conclusion" | "circular" | "ambiguous" | "shortcut" | "style_bad",
  "score": 1-5,                                              // 1=very bad, 5=excellent
  "explanation": "<one sentence>"
}

Definitions:
- "sound": reasoning is concrete, each step adds new inference, arrives at the right answer
- "wrong_conclusion": reasoning has a logical flaw, arrives at wrong answer
- "circular": just restates the story / answer without real inference
- "ambiguous": steps don't add up; conclusion is not justified
- "shortcut": skips one key inference step (e.g., jumps directly from observation to conclusion)
- "style_bad": uses fillers like "now let's think..." or just paraphrases the story
"""

USER_TMPL = """Story:
{story}

Question: {question}

Options:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Reasoning trace under review:
{reasoning}
{final}

Output the JSON verdict now."""


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse(text):
    if not text: return None
    m = _FENCE.search(text)
    if m: text = m.group(1)
    m = _OBJ.search(text)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None


def audit_one(client, rec, model="gpt-5.5", max_retries=3):
    user = USER_TMPL.format(
        story=rec["story"],
        question=rec["question"],
        opt_a=rec["options"]["A"], opt_b=rec["options"]["B"],
        opt_c=rec["options"]["C"], opt_d=rec["options"]["D"],
        reasoning=rec["reasoning"],
        final=rec["final"],
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.0, max_tokens=200, timeout=60,
            )
            obj = parse(resp.choices[0].message.content or "")
            if obj and "verdict" in obj:
                return {"qid": rec["question_id"], "task": rec["task"], "gold": rec["gold"], **obj}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/tom/raw/reasoning_traces_pilot50.jsonl")
    p.add_argument("--out", default="output/analysis/trace_audit.jsonl")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    recs = []
    with open(args.input) as f:
        for line in f:
            recs.append(json.loads(line))
    print(f"[audit] {len(recs)} traces to audit", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(audit_one, client, rec, args.model) for rec in recs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with write_lock:
                if r:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 10 == 0 or i == len(futures):
                    print(f"[audit] {i}/{len(futures)} ok={counter['ok']} fail={counter['fail']}",
                          file=sys.stderr, flush=True)
    print(f"[audit] DONE: {counter['ok']} verdicts to {out_path}")


if __name__ == "__main__":
    main()
