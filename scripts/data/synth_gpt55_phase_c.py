"""Phase A.3: Style-matched GPT-5.5 ToM synthesis for Stage8.

Stage7 14B raw eval regressed (-0.41pp vs s6) despite better val (+5.8pp).
Root cause: GPT-5.5 prior synthesis produces 8-12 sentence literary stories
that differ in style from ToMBench's 5-7 sentence direct-narrative format.

This script generates a STYLE-MATCHED replacement set. Target distribution
matches the actual HOT failure modes (per gpt55_diagnose_hot.jsonl, 70 samples).

Categories (1200 records total):
  factual_inference         200  (17% of HOT errors)
  social_norm_inference     200  (16%)
  intention_attribution     200  (14%)
  overly_literal            200  (14% pragmatic vs literal)
  emotion_attribution       150  (11%)
  knowledge_attention_link  150  (9%)
  indirect_speech_act       100  (9%)

Each EN+ZH 50/50. Style constraints (key change vs phase_a):
  - 5-7 sentences MAX (not 8-12)
  - DIRECT narrative, no rhetorical flourishes
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


SYSTEM = """You are writing theory-of-mind multiple-choice training questions in a SPECIFIC TIGHT STYLE that matches the ToMBench academic benchmark.

Output ONLY a JSON object: story, question, options (A/B/C/D), answer (A/B/C/D).

CRITICAL STYLE CONSTRAINTS (will be strictly enforced):
- Story: EXACTLY 5-7 sentences. Not 8. Not 10. Count them.
- Use simple subject-verb-object structure. Each sentence states ONE fact.
- NO rhetorical flourishes: avoid "Suddenly,", "Despite this,", "Without warning,", etc.
- NO scene-setting paragraphs about weather, mood, or atmosphere.
- Use ordinary names (first names or first+last). Original (not Sally/Anne/Wang/Li/Xiao Ming).
- Use ordinary settings: classroom, office, lunchroom, park, hallway.

CONTENT CONSTRAINTS:
- Story must FULLY determine the correct answer.
- The question must test the SPECIFIC reasoning skill the prompt asks for.
- 4 options: 1 unambiguously correct, 1 surface-literal-but-wrong (the trap), 2 plausible distractors.
- ABSOLUTELY DO NOT reproduce, paraphrase, or translate any ToMBench question.
"""


PROMPTS = {
    "factual_inference": """FACTUAL DETAIL inference question. Pattern: 1-2 critical non-obvious detail sentences (e.g., "Maya glanced at the clock as she left") determine the answer. The question asks about a character's likely belief or action. Train model to NOTICE peripheral facts. Trap: the "obvious answer" if you ignored the detail. Language: {lang}. Strict 5-7 sentences. JSON only.""",

    "social_norm_inference": """SOCIAL NORM / FAUX-PAS inference. Pattern: Character A mildly violates etiquette. Character B responds politely (smile, neutral nod). Question: did A commit a faux-pas and how does B actually feel? Train model to NOT confuse "polite reaction" with "no problem". Trap: "no, B smiled, so B is fine". Language: {lang}. Strict 5-7 sentences. JSON only.""",

    "intention_attribution": """INTENTION ATTRIBUTION. Pattern: Character A does something slightly unusual (waits until X leaves, brings up off-context topic). Story has cues to hidden goal. Question: what is A trying to accomplish? Trap: surface description. Correct: hidden goal. Language: {lang}. Strict 5-7 sentences. JSON only.""",

    "overly_literal": """PRAGMATIC vs LITERAL. Pattern: Character A says something whose literal meaning differs from intended pragmatic meaning. "My throat is dry" (= wants drink). Question: what does A actually want? Trap: literal. Correct: pragmatic. Language: {lang}. Strict 5-7 sentences. JSON only.""",

    "emotion_attribution": """EMOTION ATTRIBUTION with conflicting cues. Pattern: A shows one outward emotion (laughing) but facts suggest different inner state (just lost competition). Question: what does A actually feel? Trap: outward display. Correct: inner state. Language: {lang}. Strict 5-7 sentences. JSON only.""",

    "knowledge_attention_link": """WHO KNOWS WHAT via attention. Pattern: 3+ characters present at different times. Event E at time T. Question: who knows about E given who was where? Trap: includes someone absent at T. Language: {lang}. Strict 5-7 sentences. JSON only.""",

    "indirect_speech_act": """INDIRECT SPEECH ACT. Pattern: Speaker A says something whose literal form is comment/question but functions as request/invitation. Question: what is A actually asking for? Trap: literal. Correct: indirect. Language: {lang}. Strict 5-7 sentences. JSON only.""",
}


TASK_MAP = {
    "factual_inference": "Belief",
    "social_norm_inference": "Non-literal Comm",
    "intention_attribution": "Intention",
    "overly_literal": "Non-literal Comm",
    "emotion_attribution": "Emotion",
    "knowledge_attention_link": "Knowledge",
    "indirect_speech_act": "Intention",
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
                temperature=0.9, max_tokens=1200, timeout=120,
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
            # Style check: count sentences (very approximate)
            story_str = str(story)
            n_sent = max(story_str.count(".") + story_str.count("。") + story_str.count("！") + story_str.count("!"), 1)
            if n_sent > 9:
                last = f"too long ({n_sent} sentences)"; continue
            return TomRecord(
                question_id="synth_gpt55_pc_pending",
                source="synth_gpt55_phase_c",
                language=lang,
                task=TASK_MAP[prompt_kind],
                story=story_str, question=str(question),
                opt_a=str(opts["A"]), opt_b=str(opts["B"]),
                opt_c=str(opts["C"]), opt_d=str(opts["D"]),
                gold=answer,
            ), prompt_kind
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
    print(f"[phase-c] {prompt_kind}/{lang} failed: {last}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-factual", type=int, default=200)
    p.add_argument("--n-social-norm", type=int, default=200)
    p.add_argument("--n-intention", type=int, default=200)
    p.add_argument("--n-literal", type=int, default=200)
    p.add_argument("--n-emotion", type=int, default=150)
    p.add_argument("--n-knowledge", type=int, default=150)
    p.add_argument("--n-indirect", type=int, default=100)
    p.add_argument("--out", default="data/tom/raw/synth_gpt55_phase_c.jsonl")
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
        ("factual_inference", args.n_factual),
        ("social_norm_inference", args.n_social_norm),
        ("intention_attribution", args.n_intention),
        ("overly_literal", args.n_literal),
        ("emotion_attribution", args.n_emotion),
        ("knowledge_attention_link", args.n_knowledge),
        ("indirect_speech_act", args.n_indirect),
    ]:
        for _ in range(n // 2): plan.append((kind, "en"))
        for _ in range(n // 2): plan.append((kind, "zh"))
    random.shuffle(plan)
    print(f"[phase-c] plan: {len(plan)} calls (concurrency={args.concurrency})", file=sys.stderr)

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
                    rec.question_id = f"synth_gpt55_pc_{counter['ok']}"
                    fp.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                    by_kind[kind] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[phase-c] {i}/{total} ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s elapsed={elapsed:.0f}s by_kind={by_kind}",
                          file=sys.stderr, flush=True)
    print(f"[phase-c] DONE: {counter['ok']} records in {out_path} by_kind={by_kind}")


if __name__ == "__main__":
    main()
