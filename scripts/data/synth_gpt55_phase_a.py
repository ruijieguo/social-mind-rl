"""GPT-5.5 ToM synthesis - Phase A.1: 4 new categories targeting HOT errors.

Based on GPT-5.5 diagnosis of 70 HOT errors (where deepseek+gpt-5.5 are right
but our 14B is wrong), 70% of failures break down as:
  - factual_inference      17%  (overlooked story details)
  - social_norm_inference  16%  (faux-pas, etiquette)
  - intention_attribution  14%  (action -> hidden goal)
  - overly_literal         14%  (took surface meaning, missed indirect)
  - emotion_attribution    11%  (complex/conflicting emotions)
  - indirect_speech_act     9%
  - others                 19%

Our existing training data covers ~9% of these (false-belief). Phase A.1 fills
the gap with 1500 records:
  social_norm        400  (faux-pas + polite-reaction inversion)
  factual_detail     300  (critical non-obvious details drive answer)
  intention          400  (action + hidden goal + observer inference)
  indirect_speech    400  (hint-as-request, non-literal asks)

Each generated record passes through the same TomRecord schema + leakage check
as the original synth_gpt55.jsonl.
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
- The story must FULLY constrain the correct answer. The 4 options should be clearly distinct, with one unambiguously correct given the story.
- The question should genuinely test theory of mind (mental states, beliefs, desires, intentions, knowledge, social norms), not factual recall.
- Avoid overlapping or trivially-paraphrased options.
- Use ORIGINAL character names (NOT Sally/Anne/Tom/Wang/Li/Xiao Ming/Xiao Hong — invent fresh names).
- Use ORIGINAL settings (NOT classroom/kitchen — try office break room, art studio, gym locker room, hiking trail, etc.).
"""


PROMPTS = {
    "social_norm": """Write a SOCIAL NORM / FAUX-PAS multiple-choice question.

Setup pattern (pick one):
1. Character A says something that violates social etiquette without realizing (criticizing host's food at dinner, asking about salary in front of others, etc.). Character B responds with a POLITE reaction (forced smile, neutral nod, changing subject). Question: did a faux-pas occur and how does B actually feel?
2. Character A makes a gift / gesture that subtly signals a wrong social meaning (white flowers at a wedding in Chinese culture; a clock as a gift; etc.). Question: what does the recipient most likely think?
3. Character A's behavior shows they missed a hosting / workplace / friendship norm. Observer C must infer A's blind spot.

The key reasoning skill: the model must NOT confuse "listener didn't object" with "no faux-pas occurred". Polite reactions can mask judgment.

Constraints:
- 4 options: 1 correct (recognizes faux-pas / B's hidden feelings), 1 surface-literal wrong ("B is happy because B smiled"), 2 plausible but unsupported distractors.
- 5-9 sentences. The faux-pas must be deducible but not stated outright.

Language: {lang}

Output JSON.""",

    "factual_detail": """Write a FACTUAL DETAIL / CLOSE READING multiple-choice question about theory of mind.

Setup pattern:
- Story contains 1-2 CRITICAL non-obvious detail sentences (e.g., "She had glanced at the kitchen clock before leaving", "He noticed the cabinet door was now ajar", "She had checked her phone twice").
- These details are NOT directly asked about, but they determine the correct answer.
- The question asks about a character's belief / inference / decision.

Example: "Mara saw the kitchen clock before leaving (3:42). She returned at 4:15 to find her coffee mug missing. Earlier, Karim had said he'd leave for the gym at 4. Where does Mara now think her mug went?" Answer requires linking the timing detail to Karim's stated movement.

The key reasoning skill: the model must NOTICE peripheral facts that drive the answer, not just the headline event.

Constraints:
- 4 options: 1 correct (uses the critical detail), 1 wrong because it ignored the detail, 1 wrong because it inverted the timeline, 1 plausible distractor.
- 5-9 sentences. The critical detail should appear naturally in the narrative.

Language: {lang}

Output JSON.""",

    "intention": """Write an INTENTION ATTRIBUTION multiple-choice question.

Setup pattern:
- Character A does some action (often non-verbal, often slightly unusual: glances around, waits until others leave, brings up a topic that seems off-topic, repeats a comment with slight emphasis).
- The story provides enough context to infer A's HIDDEN GOAL.
- Question: what is A most likely trying to accomplish?

The key reasoning skill: the model must infer goal from action+context, not from literal speech. The wrong "trap" answer should be the action's surface description.

Examples of hidden goals to use:
- Salesperson's exaggerated praise → trying to upsell (not "complimenting customer")
- Coworker bringing up someone's promotion in front of boss → angling for promotion (not "informing")
- Friend asking "are you free Saturday?" before mentioning the favor → preparing the ask (not "checking availability")

Constraints:
- 4 options: 1 correct (real hidden goal), 1 surface-literal wrong (what the action looks like at face value), 1 plausibly-related-but-wrong goal, 1 distractor.
- 5-9 sentences. The hidden goal must be inferable from at least 2 cues in the story.

Language: {lang}

Output JSON.""",

    "indirect_speech": """Write an INDIRECT SPEECH ACT multiple-choice question.

Setup pattern:
- Speaker X says something that, taken literally, is a comment / observation / question, but in context it functions as a request / suggestion / complaint / invitation.
- Listener Y must infer the indirect meaning.
- Question: what is X most likely asking for / suggesting?

Examples of indirect speech acts to use:
- "My throat is so dry" → wants water/drink (NOT "weather is dry")
- "It's getting late" → wants to end the conversation/leave (NOT informing)
- "I noticed the cafe downstairs has reopened" → wants to invite the listener for coffee
- "You took the photo with the old camera?" → indirectly criticizing photo quality
- "I haven't been to your house in a while" → wants an invitation

The key reasoning skill: model must pick the INDIRECT/PRAGMATIC interpretation over the LITERAL one.

Constraints:
- 4 options: 1 correct (indirect meaning), 1 literal interpretation (the "trap"), 2 plausible but unsupported distractors.
- 5-9 sentences. Make context clearly support the indirect reading.

Language: {lang}

Output JSON.""",
}


TASK_MAP = {
    "social_norm": "Non-literal Comm",
    "factual_detail": "Belief",
    "intention": "Intention",
    "indirect_speech": "Intention",
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
            return TomRecord(
                question_id="synth_gpt55_pa_pending",
                source="synth_gpt55_phase_a",
                language=lang,
                task=TASK_MAP[prompt_kind],
                story=str(story), question=str(question),
                opt_a=str(opts["A"]), opt_b=str(opts["B"]),
                opt_c=str(opts["C"]), opt_d=str(opts["D"]),
                gold=answer,
            ), prompt_kind
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
    print(f"[phase-a] {prompt_kind}/{lang} failed: {last}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-social-norm", type=int, default=400)
    p.add_argument("--n-factual-detail", type=int, default=300)
    p.add_argument("--n-intention", type=int, default=400)
    p.add_argument("--n-indirect-speech", type=int, default=400)
    p.add_argument("--out", default="data/tom/raw/synth_gpt55_phase_a.jsonl")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    plan = []
    for kind, n in [
        ("social_norm", args.n_social_norm),
        ("factual_detail", args.n_factual_detail),
        ("intention", args.n_intention),
        ("indirect_speech", args.n_indirect_speech),
    ]:
        for _ in range(n // 2): plan.append((kind, "en"))
        for _ in range(n // 2): plan.append((kind, "zh"))
    random.shuffle(plan)
    print(f"[phase-a] plan: {len(plan)} calls (concurrency={args.concurrency})", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    by_kind = {k: 0 for k in PROMPTS}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_one, client, kind, lang, args.model) for kind, lang in plan]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with write_lock:
                if r:
                    rec, kind = r
                    rec.question_id = f"synth_gpt55_pa_{counter['ok']}"
                    fp.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                    by_kind[kind] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[phase-a] {i}/{total} ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s elapsed={elapsed:.0f}s  by_kind={by_kind}",
                          file=sys.stderr, flush=True)
    print(f"[phase-a] DONE: {counter['ok']} records in {out_path}  by_kind={by_kind}")


if __name__ == "__main__":
    main()
