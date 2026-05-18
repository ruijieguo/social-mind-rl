"""For the HOT catchable errors (where GPT-5.5 + deepseek both right but 14B wrong),
ask GPT-5.5 to explain WHY the 14B model likely got it wrong.
"""
from __future__ import annotations
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI


SYSTEM = """You are diagnosing why a Theory-of-Mind reasoning model (14B parameter Qwen3) got a multiple-choice question wrong, when both GPT-5.5 and deepseek-v4-pro got it right.

Output ONLY a JSON object:
{
  "diagnosis": "<one of these categories>",
  "explanation": "<2 sentences explaining what specific reasoning skill or detail the model missed>"
}

Categories (pick the BEST single fit):
- "second_order_belief": 2nd-order or nested mental state ('A thinks B thinks/knows X')
- "first_order_belief": 1st-order false belief, sally-anne style
- "scalar_implicature": 'almost no X' / 'most' / 'some' pragmatic counts
- "indirect_speech_act": hinting, polite indirect requests, irony
- "social_norm_inference": faux-pas / etiquette / social expectations
- "intention_attribution": predicting future actions from goals
- "emotion_attribution": inferring complex/conflicting emotions
- "desire_inference": preferences, contradictory wants
- "knowledge_attention_link": who knows what given who saw what
- "factual_inference": extracting fact from story details
- "overly_literal": model picked the literal-but-wrong interpretation
- "missed_temporal_order": event sequence / who acted first
- "letter_pattern_mistake": seems like model picked the wrong letter despite reasoning correctly
- "label_disputable": even though gpt5 + deepseek agreed, the gold seems debatable
"""


USER_TMPL = """Story:
{story}

Question: {question}

Options:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Correct answer (gold, agreed by GPT-5.5 + deepseek-v4-pro): {gold}
Our 14B model picked: {our_pred}

Diagnose what reasoning skill or detail the 14B model missed. Output JSON only."""


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


def parse_user(user_prompt):
    if user_prompt.startswith("Story:") or user_prompt.startswith("故事："):
        sm = "Story:" if user_prompt.startswith("Story:") else "故事："
        qm = "Question:" if "Question:" in user_prompt else "问题："
    else:
        return None
    _, _, after = user_prompt.partition(sm)
    story, _, qopts = after.lstrip("\n").partition(qm)
    lines = qopts.strip().split("\n")
    q = lines[0].strip()
    opts = {"A":"","B":"","C":"","D":""}
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("A."): opts["A"] = line[2:].strip()
        elif line.startswith("B."): opts["B"] = line[2:].strip()
        elif line.startswith("C."): opts["C"] = line[2:].strip()
        elif line.startswith("D."): opts["D"] = line[2:].strip()
    return story.strip(), q, opts


def diagnose_one(client, qid, lang, gold, our_pred, prompt, model="gpt-5.5"):
    parsed = parse_user(prompt)
    if not parsed: return None
    story, question, opts = parsed
    user = USER_TMPL.format(story=story, question=question,
                             opt_a=opts["A"], opt_b=opts["B"], opt_c=opts["C"], opt_d=opts["D"],
                             gold=gold, our_pred=our_pred)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.0, max_tokens=300, timeout=60,
            )
            obj = parse(resp.choices[0].message.content or "")
            if obj: return {"qid": qid, "lang": lang, "gold": gold, "our_pred": our_pred, **obj}
        except Exception as e:
            if attempt < 2: time.sleep(2 ** attempt); continue
    return None


def main():
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    hot = json.loads(Path("output/analysis/threeway_catchable_hot.json").read_text())
    eval_prompts = {}
    with open("data/tom/tombench_eval.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qid = r.get("question_id") or r.get("ground_truth", {}).get("question_id")
            user = next((m["content"] for m in r.get("messages", []) if m.get("role") == "user"), "")
            eval_prompts[qid] = user

    rng = random.Random(42)
    sample = []
    for task, errors in hot.items():
        rng.shuffle(errors)
        sample.extend([(task, e) for e in errors[:10]])
    print(f"diagnosing {len(sample)} HOT errors across {len(hot)} tasks")

    out_path = Path("output/analysis/gpt55_diagnose_hot.jsonl")
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    with out_path.open("w") as fp, ThreadPoolExecutor(max_workers=6) as ex:
        futures = {
            ex.submit(diagnose_one, client, e["qid"], e["lang"], e["gold"], e["our_pred"], eval_prompts.get(e["qid"], ""))
            : (task, e) for task, e in sample
        }
        for i, fut in enumerate(as_completed(futures), 1):
            task, e = futures[fut]
            r = fut.result()
            with write_lock:
                if r:
                    r["task"] = task
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n"); fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 10 == 0 or i == len(futures):
                    elapsed = time.time() - started
                    print(f"  {i}/{len(futures)} ok={counter['ok']} fail={counter['fail']} elapsed={elapsed:.0f}s", flush=True)
    print(f"done: wrote {counter['ok']} diagnoses to {out_path}")


if __name__ == "__main__":
    main()
