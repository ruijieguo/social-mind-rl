"""Build Hi-ToM eval (600 records, 5 ToM orders × 120 each, ~18 options each).

Source: data/tom/raw/hi_tom_gen/ToM-RL/data/cleaned_tom/raw/Hi_ToM_cleaned.csv
  Schema: deception, story_length, question_order, sample_id, story, question, choices, answer

Hi-ToM is a higher-order ToM benchmark with location MCQs that have variable
option counts (typically 18). Each question has 'choices' as comma-separated
"A. xxx, B. yyy, ..." string and 'answer' as the location text (not letter).

Output schema (jsonl):
  question_id, source=hitom, language=en, task=order_{0..4},
  story, question, options[N], gold (letter)

Usage:
  python scripts/data/build_hitom_eval.py
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from collections import Counter


SRC = Path("data/tom/raw/hi_tom_gen/ToM-RL/data/cleaned_tom/raw/Hi_ToM_cleaned.csv")


def parse_choices(choices_str: str) -> list[str]:
    """Parse "A. lettuce, B. tomato, ..." -> ["lettuce", "tomato", ...]."""
    if not choices_str:
        return []
    # Split on ", " followed by single capital letter + ". "
    parts = re.split(r",\s*(?=[A-Z]\.\s)", choices_str.strip())
    options = []
    for p in parts:
        m = re.match(r"^([A-Z])\.\s*(.+?)\s*$", p)
        if m:
            options.append(m.group(2).strip())
        else:
            options.append(p.strip())  # fallback
    return options


def text_to_letter(options: list[str], answer: str) -> str | None:
    answer = answer.strip()
    for i, o in enumerate(options):
        if o.strip().lower() == answer.lower():
            return chr(ord("A") + i)
    return None


def clean_question(q: str) -> str:
    """Strip the appended option list from the question text.

    Hi-ToM questions look like:
      "Where is the lettuce really? (blue_drawer / green_crate / ... / blue_pantry)"
    We keep only the question itself; options are presented separately.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", q).strip()


def main():
    out = Path("data/eval/hitom_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_per_order = Counter()

    with open(SRC) as f, open(out, "w") as fout:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            options = parse_choices(row.get("choices", ""))
            answer = row.get("answer", "").strip()
            order = row.get("question_order", "?")
            if not options or not answer:
                n_skipped += 1
                continue
            if len(options) > 26:
                n_skipped += 1
                continue
            letter = text_to_letter(options, answer)
            if letter is None:
                n_skipped += 1
                continue

            rec = {
                "question_id": f"hitom_order{order}_{row.get('sample_id', i)}_{i:05d}",
                "source": "hitom",
                "language": "en",
                "task": f"order_{order}",
                "story": row.get("story", ""),
                "question": clean_question(row.get("question", "")),
                "options": options,
                "gold": letter,
                "deception": row.get("deception", ""),
                "story_length": row.get("story_length", ""),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            n_per_order[order] += 1

    print(f"Wrote {n_written} records (skipped {n_skipped}) → {out}")
    print(f"Per-order: {dict(sorted(n_per_order.items()))}")
    # Distribution of option counts
    with open(out) as f:
        opt_counts = Counter(len(json.loads(line)["options"]) for line in f)
    print(f"Option-count distribution: {dict(sorted(opt_counts.items()))}")


if __name__ == "__main__":
    main()
