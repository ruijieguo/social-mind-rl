"""Convert hi_tom_3000.csv → training records for Stage 16.

The eval set uses Hi_ToM_cleaned.csv (120 stories); this source file uses
a different 600 stories per order × 5 orders = 3000 rows. Zero story
overlap with eval (verified).

We sample 240 records per order (1200 total) and convert to standard
training-message format.

Output schema (jsonl) matches data/tom/tom_train_stage14_weighted.jsonl:
  messages, ground_truth, tag=tom_mcq, source=hitom_synth, language=en,
  task=order_N, question_id

Notes:
- Hi-ToM options are 15 (variable A..O); we keep all and label as A-Z.
- The system prompt mentions "one of A, B, C, ..., O" with explicit letter set.
"""
from __future__ import annotations
import csv
import json
import random
import re
from pathlib import Path

SRC = Path("data/tom/raw/hi_tom_gen/ToM-RL/data/cleaned_tom/raw/hi_tom_3000.csv")
OUT = Path("data/tom/raw/hitom_train_1200.jsonl")
N_PER_ORDER = 240
SEED = 42


def parse_choices(s: str) -> list[str]:
    if not s:
        return []
    parts = re.split(r",\s*(?=[A-Z]\.\s)", s.strip())
    out = []
    for p in parts:
        m = re.match(r"^([A-Z])\.\s*(.+?)\s*$", p)
        out.append(m.group(2).strip() if m else p.strip())
    return out


def clean_question(q: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", q).strip()


def text_to_letter(options: list[str], answer: str) -> str | None:
    a = answer.strip().lower()
    for i, o in enumerate(options):
        if o.strip().lower() == a:
            return chr(ord("A") + i)
    return None


def build_system_prompt(n_opts: int) -> str:
    letters = ", ".join(chr(ord("A") + i) for i in range(n_opts))
    return (
        "You are a careful reader answering a multiple-choice theory-of-mind question. "
        "Read the story and the question carefully, then output ONLY your final answer "
        f"in the format \\boxed{{X}} where X is one of {letters}. "
        "Do not include any explanation, reasoning, or extra text."
    )


def build_user(story: str, question: str, options: list[str]) -> str:
    opts = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    return f"Story:\n{story}\n\nQuestion: {question}\n{opts}"


def main():
    random.seed(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # bucket by order
    rows_by_order: dict[str, list[dict]] = {}
    with open(SRC) as f:
        r = csv.DictReader(f)
        for row in r:
            rows_by_order.setdefault(row["question_order"], []).append(row)

    n_written = 0
    n_skipped = 0
    with open(OUT, "w") as fout:
        for order, rows in sorted(rows_by_order.items()):
            random.shuffle(rows)
            kept = 0
            for row in rows:
                if kept >= N_PER_ORDER:
                    break
                options = parse_choices(row.get("choices", ""))
                answer = row.get("answer", "").strip()
                if not options or not answer:
                    n_skipped += 1
                    continue
                letter = text_to_letter(options, answer)
                if letter is None:
                    n_skipped += 1
                    continue
                question = clean_question(row.get("question", ""))
                if not question:
                    n_skipped += 1
                    continue
                rec = {
                    "messages": [
                        {"role": "system", "content": build_system_prompt(len(options))},
                        {"role": "user", "content": build_user(row["story"], question, options)},
                    ],
                    "ground_truth": letter,
                    "tag": "tom_mcq",
                    "source": "hitom_synth",
                    "language": "en",
                    "task": f"order_{order}",
                    "question_id": f"hitom_train_order{order}_{row['sample_id']}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                n_written += 1
            print(f"order_{order}: kept {kept}")
    print(f"\nWrote {n_written} records (skipped {n_skipped}) → {OUT}")


if __name__ == "__main__":
    main()
