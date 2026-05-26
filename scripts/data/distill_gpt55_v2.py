"""GPT-5.5 distillation v2 — 3-sample voting + ontology-aware system prompt.

Improvements over distill_gpt55.py:
1. Solve step uses **3 samples (T=0.4)** with majority voting; require ≥2/3 to match
   gold (filters GPT-5.5 own noise, raises retain quality at modest cost).
2. Optional --inject-ontology: prepend a fine-grained emotion / belief ontology to
   the SOLVE step's system prompt, so GPT-5.5's reasoning shows discriminating
   features model can imitate.
3. Error pool: by default uses **v3.4 errors** (smaller, higher-quality target),
   not the historical v3.3 errors that v3.4 already saw.

Usage:
  python scripts/data/distill_gpt55_v2.py \\
    --eval-results output/eval/stage18_ckpt30_emobench.json \\
    --eval-data data/eval/emobench_eval.jsonl \\
    --tasks EU_emotion EU_cause EA \\
    --inject-ontology data/distill/emotion_ontology.txt \\
    --paraphrase-multiplier 1 \\
    --out data/tom/raw/distill_v2_emobench.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openai import OpenAI


SYS_PARAPHRASE = """You are a careful paraphraser. Given a multiple-choice question with a story, your job is to rewrite ONLY the story while preserving:
- The same number of characters / actors as the original
- The same task structure (who knows what, who saw what, the same belief/emotion/social dynamic)
- The same reasoning pattern needed to answer correctly
- **The same correct answer (CRITICAL — your paraphrase must NOT change which option is correct)**

What you CAN change:
- Character names (use different culturally-appropriate names)
- Settings (different location/place but same kind of social environment)
- Concrete objects (different specific items but same role)
- Surface wording

What you MUST PRESERVE:
- The emotional valence (positive→positive, frustrated→frustrated)
- The social dynamic (who has higher/lower status, who has more info)
- The cause-effect chain (why the protagonist would feel/think the way the gold answer indicates)
- Specific details that justify the gold answer (don't add or remove key facts)

DO NOT change the question or the options. ONLY paraphrase the story.

Before outputting, sanity-check: given your paraphrase, does the gold-answer option still uniquely fit? If not, rewrite.

Output strict JSON:
{
  "story": "<paraphrased story, 1-4 sentences>"
}
No extra text."""


SYS_SOLVE_BASE = """You are a careful theory-of-mind / emotion / social reasoner. Given a story and a multiple-choice question, write 2-4 short sentences of step-by-step reasoning, then output your answer on the last line as \\boxed{X} where X is one of the option letters.

Keep reasoning concise — 50-150 words — then the boxed answer."""


def build_paraphrase_user(story, question, options, gold_letter, lang):
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    return (
        f"Original story:\n{story}\n\n"
        f"Question: {question}\n"
        f"{opts_block}\n"
        f"Correct answer: {gold_letter}\n\n"
        f"Language: {lang}. Output paraphrased story only in JSON."
    )


def build_solve_user(story, question, options, lang):
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    if lang == "zh":
        return f"故事：\n{story}\n\n问题：{question}\n{opts_block}"
    return f"Story:\n{story}\n\nQuestion: {question}\n{opts_block}"


def parse_paraphrase_json(text):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return (json.loads(m.group(0)).get("story") or "").strip() or None
    except Exception:
        return None


def extract_boxed(text):
    m = re.search(r"\\boxed\{([A-Z])\}", text)
    return m.group(1) if m else None


def to_training_record(*, story, question, options, gold_letter, source, task, language, qid):
    n = len(options)
    letters = ", ".join(chr(ord("A") + i) for i in range(n))
    sys_p = (
        "You are a careful reader answering a multiple-choice question. "
        "Read the story and the question carefully, then output ONLY your final answer "
        f"in the format \\boxed{{X}} where X is one of {letters}. "
        "Do not include any explanation, reasoning, or extra text."
    )
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    user_p = (f"故事：\n{story}\n\n问题：{question}\n{opts_block}" if language == "zh"
              else f"Story:\n{story}\n\nQuestion: {question}\n{opts_block}")
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
        ],
        "ground_truth": gold_letter,
        "tag": "tom_mcq",
        "source": source,
        "language": language,
        "task": task,
        "question_id": qid,
    }


def extract_record_fields(eval_rec):
    story = eval_rec.get("story", "")
    question = eval_rec.get("question", "")
    lang = eval_rec.get("language", "en")
    task = eval_rec.get("task", "?")
    if "opt_a" in eval_rec:
        options = [eval_rec[k] for k in ("opt_a", "opt_b", "opt_c", "opt_d")]
    else:
        options = eval_rec.get("options", [])
    gold = eval_rec.get("gold", "")
    if gold and len(gold) == 1 and gold.isalpha():
        gold_letter = gold.upper()
    elif isinstance(gold, int):
        gold_letter = chr(ord("A") + gold)
    else:
        gold_letter = ""
    return story, question, options, gold_letter, lang, task


def collect_errors(eval_results_fn, eval_data_fn, tasks):
    err_qids = set()
    for r in json.load(open(eval_results_fn)):
        if tasks and r.get("task") not in tasks:
            continue
        if not r.get("correct"):
            err_qids.add(r["question_id"])
    eval_map = {}
    for line in open(eval_data_fn):
        r = json.loads(line)
        if r["question_id"] in err_qids and (not tasks or r.get("task") in tasks):
            eval_map[r["question_id"]] = r
    return list(eval_map.values())


def distill_one(client, model, eval_rec, variant_idx, n_samples, vote_threshold, ontology_text):
    story, question, options, gold_letter, lang, task = extract_record_fields(eval_rec)
    if not options or not gold_letter:
        return None

    sys_solve = (ontology_text + "\n\n" + SYS_SOLVE_BASE) if ontology_text else SYS_SOLVE_BASE

    # Step 0 (NEW): pre-verify on ORIGINAL story — if GPT-5.5 itself can't get
    # gold on the original story, this question isn't a good distill candidate.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_solve},
                {"role": "user", "content": build_solve_user(story, question, options, lang)},
            ],
            temperature=0.0, top_p=1.0, max_tokens=2048, timeout=120,
        )
        pre_pred = extract_boxed(resp.choices[0].message.content or "")
        if pre_pred != gold_letter:
            return None  # GPT-5.5 doesn't agree with gold on original — drop
    except Exception:
        return None

    # Step 1: paraphrase
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS_PARAPHRASE},
                {"role": "user", "content": build_paraphrase_user(story, question, options, gold_letter, lang)},
            ],
            temperature=0.85, top_p=0.95, max_tokens=1024, timeout=60,
        )
        para_text = resp.choices[0].message.content or ""
    except Exception:
        return None
    para_story = parse_paraphrase_json(para_text)
    if not para_story:
        return None

    # Step 2: 3-sample voting solve on PARAPHRASED story
    preds = []
    for _ in range(n_samples):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_solve},
                    {"role": "user", "content": build_solve_user(para_story, question, options, lang)},
                ],
                temperature=0.4, top_p=0.9, max_tokens=2048, timeout=120,
            )
            txt = resp.choices[0].message.content or ""
            p = extract_boxed(txt)
            if p is not None:
                preds.append(p)
        except Exception:
            continue

    if not preds:
        return None
    cnt = Counter(preds)
    top_letter, top_count = cnt.most_common(1)[0]
    if top_letter != gold_letter or top_count < vote_threshold:
        return None  # vote failed — paraphrase drifted from gold

    return to_training_record(
        story=para_story, question=question, options=options,
        gold_letter=gold_letter, source="gpt55_distill_v2",
        task=task, language=lang,
        qid=f"distillv2_{eval_rec['question_id']}_v{variant_idx}",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-results", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--paraphrase-multiplier", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=3, help="samples per item for voting")
    p.add_argument("--vote-threshold", type=int, default=2, help="min agreeing samples")
    p.add_argument("--inject-ontology", default=None,
                   help="path to ontology text file; injected into solve system prompt")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    ontology_text = None
    if args.inject_ontology:
        ontology_text = Path(args.inject_ontology).read_text().strip()
        print(f"Injecting ontology ({len(ontology_text)} chars) into solve system prompt")

    task_filter = set(args.tasks) if args.tasks else None
    errors = collect_errors(args.eval_results, args.eval_data, task_filter)
    print(f"Found {len(errors)} unique error records (tasks={task_filter})")
    if args.limit:
        errors = errors[: args.limit]

    work = [(rec, v) for rec in errors for v in range(args.paraphrase_multiplier)]
    print(f"Submitting {len(work)} distill tasks "
          f"(samples={args.n_samples}, vote≥{args.vote_threshold}, concurrency={args.concurrency})")

    written = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(distill_one, client, args.model, rec, v,
                          args.n_samples, args.vote_threshold, ontology_text): (rec, v)
                for rec, v in work}
        for i, fut in enumerate(as_completed(futs)):
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                written.append(rec)
            if (i + 1) % 30 == 0:
                print(f"  progress {i+1}/{len(work)}  retained {len(written)}", flush=True)

    print(f"\nWrote {len(written)} records → {args.out}")
    print(f"Retain rate: {len(written)/max(len(work),1):.1%}")


if __name__ == "__main__":
    main()
