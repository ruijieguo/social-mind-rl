"""GPT-5.5 ToM training data synthesis.

Per-record cost is much higher than deepseek-v4-flash (~$0.05 vs $0.001),
but the audit shows synth/synth_zh from deepseek-flash are already clean
(100% label correct, 96% high training_value). The bottleneck is volume,
not quality, since we're dropping ~2700 records of ExploreToM.

Strategy: Use GPT-5.5 only to fill the False Belief gap left by ExploreToM
removal. False Belief is the easiest task to define cleanly (Sally-Anne
template) and the most volume we lose.

Generates ~1500 records targeting:
  - False Belief (1st and 2nd order) — replace lost ExploreToM
  - Plus a small amount of Knowledge (scalar implicature) since that's
    where deepseek beats us by 9pp on full 5718.

Output: data/tom/raw/synth_gpt55.jsonl
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

from scripts.data.schema import TomRecord


SYSTEM = """You are an expert writer of theory-of-mind multiple-choice training questions.

Output ONLY a single valid JSON object with keys: story, question, options (object with A/B/C/D), answer (one of A/B/C/D).

Hard requirements:
- ABSOLUTELY DO NOT reproduce, paraphrase, or translate any question from ToMBench (Chen et al. ACL 2024) or any other published ToM benchmark.
- The story must FULLY constrain the correct answer. Avoid ambiguity. The 4 options should be clearly distinct, with one unambiguously correct given the story.
- The question should genuinely test theory of mind (mental states, beliefs, desires, intentions, knowledge), not factual recall.
- Avoid overlapping or trivially-paraphrased options.
"""


PROMPTS = {
    "false_belief_1st": """Write a FIRST-ORDER False Belief task.

Setup template:
- Two characters X and Y are in a room. They both observe object O placed in container A.
- One of them (say Y) leaves the room.
- The remaining character (X) moves O to container B without Y observing.
- Y returns.

Question: where will Y look for O?  (Correct answer: A — Y still believes O is in A.)

Constraints:
- Use original character names (NOT Sally/Anne, Wang/Li, Xiao Ming/Xiao Hong — invent fresh names).
- Use original setting (NOT a kitchen — try lab, library, workshop, etc.).
- Use 4 plausible container choices for the options. The correct answer should be the original container A; one option should be B (the new location, which the model would say if it confused 1st-order vs reality); two options should be other containers in the same scene as plausible distractors.
- 5-8 sentences. Clear, unambiguous.

The 4 options should be ~A/B/C/D containing the correct A (original) and B (new) and 2 plausible-but-not-mentioned distractors.

Language: {lang}

Output JSON.""",

    "false_belief_2nd": """Write a SECOND-ORDER False Belief task.

Setup template:
- Three characters X, Y, Z. Y witnesses an event E. Z does not witness E.
- X knows that Y witnessed E and that Z did not.
- Now ask: what does X think Z believes?

Or: X moves an object while Y watches; Y leaves; Z then secretly moves it again. Question: what does X think Y believes about the location?

Constraints:
- Original character names (NOT Sally/Anne/Tom, NOT Wang/Li/Xiao Ming).
- Original setting.
- Multi-clause logical chain, but the answer must still be unambiguous.
- 6-10 sentences.
- 4 options, one clearly correct.

Language: {lang}

Output JSON.""",

    "knowledge_scalar": """Write a Knowledge task involving scalar implicature with a numeric twist.

Setup template:
- Speaker A tells listener B: "There are N total items. Most are X, some are Y, almost no Z."
- B then verifies one count (e.g., "B finds only 4 items of type Y").
- Question: based on A's description, before B verified, what number of X did B estimate?

The answer must use scalar implicature:
- "almost no" Z means Z ≈ 0-3 (idiomatic, not literal "0")
- "some" Y is what B verified (e.g., 4)
- "most" X = total − verified Y − implied tiny Z

Example: total=20, "most are burgers, some are sandwiches, almost no salad". B finds 4 sandwiches. Implied: ~1 salad. So estimated burgers = 20 − 4 − 1 = 15.

Constraints:
- Pick original total numbers (not 50, not 20 — try 30, 40, 60, 75, etc.)
- Pick original setting (school cafeteria, library, parking lot, art studio, etc.)
- The 4 options should be:
  - The pragmatic-implicature answer (correct, uses "almost no" → ~1)
  - The literal-arithmetic answer (total − Y, ignores "almost no" → wrong)
  - Two distractors (e.g., total − Y − 5, or total / 2)
- Make sure only the pragmatic answer is fully correct.

Language: {lang}

Output JSON.""",
}


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_response(text):
    if not text: return None
    m = _FENCE.search(text)
    if m: text = m.group(1)
    m = _OBJ.search(text)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def call_one(client, prompt_kind, lang, model="gpt-5.5", max_retries=3):
    user = PROMPTS[prompt_kind].format(lang=lang)
    last = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.9, max_tokens=1500, timeout=120,
            )
            content = resp.choices[0].message.content or ""
            obj = parse_response(content)
            if not obj:
                last = f"parse fail: {content[:100]!r}"; continue
            try:
                story, question, opts, answer = obj["story"], obj["question"], obj["options"], obj["answer"]
            except (KeyError, TypeError):
                last = f"missing keys: {list(obj.keys())}"; continue
            if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
                last = "bad options"; continue
            answer = str(answer).strip().upper()
            if answer not in {"A","B","C","D"}:
                last = f"bad answer {answer!r}"; continue
            task_map = {"false_belief_1st": "False Belief", "false_belief_2nd": "False Belief", "knowledge_scalar": "Knowledge"}
            return TomRecord(
                question_id="synth_gpt55_pending",
                source="synth_gpt55",
                language=lang,
                task=task_map[prompt_kind],
                story=str(story), question=str(question),
                opt_a=str(opts["A"]), opt_b=str(opts["B"]),
                opt_c=str(opts["C"]), opt_d=str(opts["D"]),
                gold=answer,
            )
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
    print(f"[gpt55-synth] {prompt_kind}/{lang} failed: {last}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-fb-1st", type=int, default=600, help="false_belief 1st-order")
    p.add_argument("--n-fb-2nd", type=int, default=400, help="false_belief 2nd-order")
    p.add_argument("--n-scalar", type=int, default=400, help="knowledge scalar implicature")
    p.add_argument("--out", default="data/tom/raw/synth_gpt55.jsonl")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    plan = []
    for kind, n in [("false_belief_1st", args.n_fb_1st), ("false_belief_2nd", args.n_fb_2nd), ("knowledge_scalar", args.n_scalar)]:
        for _ in range(n // 2): plan.append((kind, "en"))
        for _ in range(n // 2): plan.append((kind, "zh"))
    random.shuffle(plan)
    print(f"[gpt55-synth] plan: {len(plan)} calls", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_one, client, kind, lang, args.model) for kind, lang in plan]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with write_lock:
                if rec:
                    rec.question_id = f"synth_gpt55_{counter['ok']}"
                    fp.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[gpt55-synth] {i}/{total} ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s elapsed={elapsed:.0f}s",
                          file=sys.stderr, flush=True)
    print(f"[gpt55-synth] DONE: {counter['ok']} records in {out_path}")


if __name__ == "__main__":
    main()
