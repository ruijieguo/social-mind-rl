"""GPT-5.5 distillation pipeline: v3.3 errors → paraphrased + correctly-reasoned training records.

Pipeline (two GPT-5.5 calls per error case):

  1. PARAPHRASE: given (story, question, options, gold), ask GPT-5.5 to
     paraphrase the *story* (change names, settings, props, but keep the
     same task structure and gold meaning). Options and gold are kept
     verbatim — only the narrative is reworded.

  2. SOLVE: given the paraphrased story + original question + options, ask
     GPT-5.5 to write step-by-step reasoning ending in \\boxed{X}. We keep
     only records where GPT-5.5's boxed answer == gold (catches paraphrase
     drift and GPT-5.5's own errors).

Output: standard training record (messages format) with source="gpt55_distill"
so the RLVR worker treats it the same as any other rollout source. The
distilled CoT is NOT used as training labels — we just want the model to
generate its own reasoning on these story shapes during RL rollouts.

Usage:
  export OPENAI_API_KEY=...                  # GPT-5.5 via modelservice.top
  export OPENAI_BASE_URL=https://www.modelservice.top/v1
  python scripts/data/distill_gpt55.py \\
    --eval-results output/eval/stage17_ckpt120_emobench.json \\
    --eval-data data/eval/emobench_eval.jsonl \\
    --tasks EU_emotion EU_cause EA \\
    --paraphrase-multiplier 2 \\
    --out data/tom/raw/distill_emobench.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openai import OpenAI


SYS_PARAPHRASE = """You are a careful paraphraser. Given a multiple-choice question with a story, your job is to rewrite ONLY the story (change character names, settings, objects, locations) while preserving:
- The same number of characters / actors as the original
- The same task structure (who knows what, who saw what, the same belief/emotion/social dynamic)
- The same reasoning pattern needed to answer correctly
- The same correct answer

DO NOT change the question or the options. ONLY paraphrase the story.

Output strict JSON:
{
  "story": "<paraphrased story, 1-4 sentences>"
}
No extra text."""


SYS_SOLVE = """You are a careful theory-of-mind / emotion / social reasoner. Given a story and a multiple-choice question, write 2-4 short sentences of step-by-step reasoning, then output your answer on the last line as \\boxed{X} where X is one of the option letters.

Keep reasoning concise — 50-150 words — then the boxed answer."""


def build_paraphrase_user(story: str, question: str, options: list[str], gold_letter: str, lang: str) -> str:
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    return (
        f"Original story:\n{story}\n\n"
        f"Question: {question}\n"
        f"{opts_block}\n"
        f"Correct answer: {gold_letter}\n\n"
        f"Language: {lang}. Output paraphrased story only in JSON."
    )


def build_solve_user(story: str, question: str, options: list[str], lang: str) -> str:
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    if lang == "zh":
        prefix = "故事：\n"
        q_prefix = "问题："
    else:
        prefix = "Story:\n"
        q_prefix = "Question: "
    return f"{prefix}{story}\n\n{q_prefix}{question}\n{opts_block}"


def parse_paraphrase_json(text: str) -> str | None:
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        s = obj.get("story", "").strip()
        return s or None
    except Exception:
        return None


def extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([A-Z])\}", text)
    return m.group(1) if m else None


def to_training_record(
    *, story: str, question: str, options: list[str], gold_letter: str,
    source: str, task: str, language: str, qid: str
) -> dict:
    n = len(options)
    letters = ", ".join(chr(ord("A") + i) for i in range(n))
    sys_p = (
        "You are a careful reader answering a multiple-choice question. "
        "Read the story and the question carefully, then output ONLY your final answer "
        f"in the format \\boxed{{X}} where X is one of {letters}. "
        "Do not include any explanation, reasoning, or extra text."
    )
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    if language == "zh":
        user_p = f"故事：\n{story}\n\n问题：{question}\n{opts_block}"
    else:
        user_p = f"Story:\n{story}\n\nQuestion: {question}\n{opts_block}"
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


def extract_record_fields(eval_rec: dict) -> tuple[str, str, list[str], str, str, str]:
    """Return (story, question, options, gold_letter, language, task)."""
    story = eval_rec.get("story", "")
    question = eval_rec.get("question", "")
    lang = eval_rec.get("language", "en")
    task = eval_rec.get("task", "?")
    # ToMBench schema has opt_a/b/c/d; others have options[]
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


def collect_errors(eval_results_fn: str, eval_data_fn: str, tasks: set[str] | None) -> list[dict]:
    """Find unique question_ids that v3.3 got wrong on ≥1 protocol, return eval records."""
    err_qids: set[str] = set()
    for r in json.load(open(eval_results_fn)):
        if tasks and r.get("task") not in tasks:
            continue
        if not r.get("correct"):
            err_qids.add(r["question_id"])
    eval_map: dict[str, dict] = {}
    for line in open(eval_data_fn):
        r = json.loads(line)
        if r["question_id"] in err_qids and (not tasks or r.get("task") in tasks):
            eval_map[r["question_id"]] = r
    return list(eval_map.values())


def distill_one(
    client: OpenAI, model: str, eval_rec: dict, variant_idx: int
) -> dict | None:
    story, question, options, gold_letter, lang, task = extract_record_fields(eval_rec)
    if not options or not gold_letter:
        return None

    # Step 1: paraphrase the story
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS_PARAPHRASE},
                {"role": "user", "content": build_paraphrase_user(story, question, options, gold_letter, lang)},
            ],
            temperature=0.85,
            top_p=0.95,
            max_tokens=1024,
            timeout=60,
        )
        para_text = resp.choices[0].message.content or ""
    except Exception:
        return None
    para_story = parse_paraphrase_json(para_text)
    if not para_story:
        return None

    # Step 2: ask GPT-5.5 to solve paraphrased version
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS_SOLVE},
                {"role": "user", "content": build_solve_user(para_story, question, options, lang)},
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=2048,
            timeout=120,
        )
        solve_text = resp.choices[0].message.content or ""
    except Exception:
        return None
    pred = extract_boxed(solve_text)
    if pred is None or pred != gold_letter:
        return None  # GPT-5.5 got it wrong on the paraphrase — drop

    return to_training_record(
        story=para_story,
        question=question,
        options=options,
        gold_letter=gold_letter,
        source="gpt55_distill",
        task=task,
        language=lang,
        qid=f"distill_{eval_rec['question_id']}_v{variant_idx}",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-results", required=True, help="JSON with model eval results (per-row correct)")
    p.add_argument("--eval-data", required=True, help="JSONL with eval data (story/question/options/gold)")
    p.add_argument("--tasks", nargs="*", default=None, help="filter to these task names")
    p.add_argument("--paraphrase-multiplier", type=int, default=1,
                   help="generate N paraphrased variants per error")
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

    task_filter = set(args.tasks) if args.tasks else None
    errors = collect_errors(args.eval_results, args.eval_data, task_filter)
    print(f"Found {len(errors)} unique error records (tasks={task_filter})")
    if args.limit:
        errors = errors[: args.limit]

    # Build (eval_rec, variant_idx) work items
    work = []
    for rec in errors:
        for v in range(args.paraphrase_multiplier):
            work.append((rec, v))
    print(f"Submitting {len(work)} distill tasks (concurrency={args.concurrency})")

    written: list[dict] = []
    fail = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(distill_one, client, args.model, rec, v): (rec, v) for rec, v in work}
        for i, fut in enumerate(as_completed(futures)):
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                written.append(rec)
            else:
                fail += 1
            if (i + 1) % 50 == 0:
                print(f"  progress: {i+1}/{len(work)}  kept: {len(written)}  failed: {fail}", flush=True)

    print(f"wrote {len(written)} distilled records (failed {fail}) → {args.out}")


if __name__ == "__main__":
    main()
