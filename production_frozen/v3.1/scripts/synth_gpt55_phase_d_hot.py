"""
Track C: HOT-targeted training-data synthesis via GPT-5.5.

For each task category in the 492 HOT questions (where stage 8 is wrong but
both gpt-5.5 and deepseek are right), produce N similar-pattern training
questions in different scenarios. The aim is to cover the reasoning patterns
stage 8 misses, without leaking eval data.

Pipeline:
  output/analysis/hot_questions.jsonl  (already extracted)
  -> data/tom/raw/synth_gpt55_phase_d_hot.jsonl

Each output record matches the standard ToMBench training format:
  question_id, source, language, task, story, question, opt_a..opt_d, gold
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict


SYSTEM_PROMPT = """You are a Theory of Mind question generator. Given several reference questions
of a specific task type, produce ONE new question that:
1. Has the SAME task type and reasoning pattern.
2. Uses ENTIRELY different characters, settings, and props.
3. Is in {language}.
4. Has 4 plausible options (A/B/C/D), exactly ONE clearly correct.
5. Length matches references (typically 100-250 chars in story).
6. Does NOT copy any phrase from the reference questions.

Output strict JSON with fields:
  story, question, opt_a, opt_b, opt_c, opt_d, gold
where gold is "A", "B", "C", or "D".
NO additional text outside the JSON object."""


USER_TEMPLATE = """Reference questions of task type "{task}" (DO NOT COPY):

{refs}

Now produce ONE NEW question of the same type, in {language}.
Different characters, different setting. Output JSON only."""


def format_ref(rec, idx):
    return (
        f"--- Reference {idx} ---\n"
        f"Story: {rec['story']}\n"
        f"Question: {rec['question']}\n"
        f"A. {rec['opt_a']}\nB. {rec['opt_b']}\nC. {rec['opt_c']}\nD. {rec['opt_d']}\n"
        f"Gold: {rec['gold']}"
    )


def gen_one(client, model, task, language, refs):
    prompt = USER_TEMPLATE.format(
        task=task, language=language,
        refs="\n\n".join(format_ref(r, i + 1) for i, r in enumerate(refs)),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": prompt},
        ],
        temperature=0.95,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content.strip()
    obj = json.loads(text)
    for k in ("story", "question", "opt_a", "opt_b", "opt_c", "opt_d", "gold"):
        assert k in obj, f"missing {k}"
    assert obj["gold"] in {"A", "B", "C", "D"}, f"bad gold: {obj['gold']}"
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/analysis/hot_questions.jsonl")
    ap.add_argument("--output", default="data/tom/raw/synth_gpt55_phase_d_hot.jsonl")
    ap.add_argument("--model", default="gpt-5.5-1106")
    ap.add_argument("--n-per-task", type=int, default=180)
    ap.add_argument("--refs-per-prompt", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        from openai import OpenAI
    except ImportError:
        print("openai package missing"); sys.exit(1)

    hot = [json.loads(l) for l in open(args.input)]
    by_task = defaultdict(list)
    for r in hot:
        by_task[r["task"]].append(r)

    print("HOT pattern samples per task:")
    for t, lst in by_task.items():
        en = sum(1 for r in lst if r.get("language", "en") == "en")
        zh = len(lst) - en
        print(f"  {t:<18}: total={len(lst)} EN={en} ZH={zh}")

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if Path(args.output).exists():
        for line in open(args.output):
            try: existing_ids.add(json.loads(line)["question_id"])
            except Exception: pass
    print(f"Existing records: {len(existing_ids)}")

    out_f = open(args.output, "a")
    t0 = time.time()
    total_attempts = total_success = 0

    for task, refs in sorted(by_task.items()):
        en_refs = [r for r in refs if r.get("language", "en") == "en"]
        zh_refs = [r for r in refs if r.get("language", "en") == "zh"]

        for lang_code, lang_refs, lang_name in [
            ("en", en_refs, "English"),
            ("zh", zh_refs, "Chinese"),
        ]:
            if not lang_refs:
                continue
            target_n = args.n_per_task // 2
            done_n = 0
            for i in range(target_n * 2):
                if done_n >= target_n:
                    break
                qid = f"synth_d_hot__{task.replace(' ', '_').lower()}__{lang_code}__{i:04d}"
                if qid in existing_ids:
                    continue
                refs_sampled = random.sample(lang_refs, min(args.refs_per_prompt, len(lang_refs)))
                total_attempts += 1
                for retry in range(args.max_retries + 1):
                    try:
                        obj = gen_one(client, args.model, task, lang_name, refs_sampled)
                        rec = {
                            "question_id": qid,
                            "source": "synth_gpt55_phase_d_hot",
                            "language": lang_code,
                            "task": task,
                            **obj,
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        total_success += 1
                        done_n += 1
                        break
                    except Exception as e:
                        if retry == args.max_retries:
                            print(f"  ! {qid} failed after retries: {e}", flush=True)
                        else:
                            time.sleep(2 + retry * 2)
                if total_attempts % 10 == 0:
                    elapsed = time.time() - t0
                    rate = total_success / elapsed * 60 if elapsed > 0 else 0
                    print(f"  attempts={total_attempts} success={total_success} elapsed={elapsed:.0f}s rate={rate:.1f}/min", flush=True)
        print(f"Done task {task}", flush=True)

    out_f.close()
    print(f"\nDONE: attempts={total_attempts} success={total_success}")


if __name__ == "__main__":
    main()
