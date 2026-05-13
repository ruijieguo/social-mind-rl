"""Translate English training data to Chinese via deepseek-v4-flash.

Why: baseline audit showed 100% of training data is English while ToMBench
has ~50% Chinese. Training on English-only will not improve (and may
regress) Chinese ToMBench accuracy.

Approach: take a random ~1500 English records, translate story + question
+ options via deepseek-v4-flash, rewrite the user message with Chinese
labels (故事：/ 问题：) and emit them as new zh records.

Writes streamingly to data/tom/raw/zh_translated.jsonl so a crash leaves
partial progress usable. The caller (merge_and_dedupe) will pick it up.
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
from typing import Optional

import jsonlines
from openai import OpenAI


TRANSLATE_SYSTEM = (
    "You are translating multiple-choice theory-of-mind questions from English to "
    "simplified Chinese for model training. Preserve the exact JSON schema; translate "
    "story, question and all four options faithfully without adding or removing content. "
    "Keep people's names consistent (you may use pinyin or common Chinese equivalents). "
    "Output ONE json object with keys: story, question, options (A,B,C,D), answer (A/B/C/D)."
)


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")

_USER_RE = re.compile(
    r"Story:\s*(?P<story>.+?)\n"
    r"Question:\s*(?P<q>.+?)\n"
    r"A\.\s*(?P<a>.+?)\n"
    r"B\.\s*(?P<b>.+?)\n"
    r"C\.\s*(?P<c>.+?)\n"
    r"D\.\s*(?P<d>.+?)$",
    re.DOTALL,
)

SYSTEM_PROMPT_DIRECT_ZH_KEEP = None  # Reuse the same English system; training loads both.


def parse_english_user(content: str) -> Optional[dict]:
    m = _USER_RE.match(content.strip())
    if not m:
        return None
    return {
        "story": m.group("story").strip(),
        "question": m.group("q").strip(),
        "A": m.group("a").strip(),
        "B": m.group("b").strip(),
        "C": m.group("c").strip(),
        "D": m.group("d").strip(),
    }


def parse_translation_response(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    m = _OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    try:
        story = obj["story"]
        question = obj["question"]
        opts = obj["options"]
        answer = obj["answer"]
    except (KeyError, TypeError):
        return None
    if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
        return None
    answer = str(answer).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return None
    return {
        "story": str(story),
        "question": str(question),
        "A": str(opts["A"]),
        "B": str(opts["B"]),
        "C": str(opts["C"]),
        "D": str(opts["D"]),
        "answer": answer,
    }


def translate_once(client: OpenAI, en: dict, gold: str, model: str, max_retries: int = 3) -> Optional[dict]:
    prompt_en_json = json.dumps({
        "story": en["story"],
        "question": en["question"],
        "options": {"A": en["A"], "B": en["B"], "C": en["C"], "D": en["D"]},
        "answer": gold,
    }, ensure_ascii=False)
    user = (
        f"Translate the following to simplified Chinese as a JSON object:\n\n"
        f"{prompt_en_json}"
    )
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TRANSLATE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=1200,
                timeout=60,
            )
            content = resp.choices[0].message.content or ""
            parsed = parse_translation_response(content)
            if parsed is None:
                last_err = f"parse failed; content[:100]={content[:100]!r}"
                continue
            if parsed["answer"] != gold:
                last_err = f"gold letter changed during translation ({gold} -> {parsed['answer']})"
                continue
            return parsed
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    print(f"[translate] failed after {max_retries} retries: {last_err}", file=sys.stderr, flush=True)
    return None


def build_zh_user_prompt(story: str, question: str, a: str, b: str, c: str, d: str) -> str:
    return (
        f"故事：\n{story}\n\n"
        f"问题：{question}\n"
        f"A. {a}\nB. {b}\nC. {c}\nD. {d}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--source-file", default="data/tom/tom_train.jsonl",
                   help="EN training JSONL to pick source records from")
    p.add_argument("--out", default="data/tom/raw/zh_translated.jsonl")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    rng = random.Random(args.seed)
    en_records = [r for r in jsonlines.open(args.source_file) if r.get("language", "en") == "en"]
    print(f"[translate] source pool: {len(en_records)} English records", file=sys.stderr, flush=True)

    # Sample n distinct source records (need to parse user prompt first).
    parsed_pool: list[tuple[dict, str, str, str]] = []
    for r in en_records:
        user_msg = next((m for m in r["messages"] if m["role"] == "user"), None)
        if not user_msg:
            continue
        en = parse_english_user(user_msg["content"])
        if en is None:
            continue
        parsed_pool.append((en, r["ground_truth"], r["source"], r["question_id"]))
    print(f"[translate] parseable records: {len(parsed_pool)}", file=sys.stderr, flush=True)

    sample = rng.sample(parsed_pool, k=min(args.n, len(parsed_pool)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()

    with out.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(translate_once, client, en, gold, args.model, args.max_retries): (en, gold, src, qid)
            for (en, gold, src, qid) in sample
        }
        total = len(futures)
        for i, f in enumerate(as_completed(futures), 1):
            en, gold, src, qid = futures[f]
            zh = f.result()
            with write_lock:
                if zh is not None:
                    zh_user = build_zh_user_prompt(
                        zh["story"], zh["question"],
                        zh["A"], zh["B"], zh["C"], zh["D"],
                    )
                    system_content = (
                        "You are a careful reader answering a multiple-choice theory-of-mind question. "
                        "Read the story and the question carefully, then output ONLY your final answer "
                        "in the format \\boxed{X} where X is one of A, B, C, D. "
                        "Do not include any explanation, reasoning, or extra text."
                    )
                    new_rec = {
                        "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": zh_user},
                        ],
                        "ground_truth": zh["answer"],
                        "tag": "tom_mcq",
                        "source": f"{src}_zh",
                        "language": "zh",
                        "task": "Other",
                        "question_id": f"{qid}_zh",
                    }
                    fp.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(
                        f"[translate] {i}/{total} done | ok={counter['ok']} fail={counter['fail']} | "
                        f"{rate:.1f} req/s | elapsed={elapsed:.0f}s",
                        file=sys.stderr, flush=True,
                    )

    print(
        f"[translate] FINAL: wrote {counter['ok']} zh records to {out} (failed: {counter['fail']})",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
