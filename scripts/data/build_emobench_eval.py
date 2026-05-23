"""Build EmoBench (mixed: 400 EU emotion-recognition + 400 EU cause + 400 EA action).

Schema differs between subsets:
- emotional_application (EA): {qid, language, category, question type, scenario,
                                subject, choices, label}  → 1 question per row
- emotional_understanding (EU): {qid, language, coarse_category, finegrained_category,
                                  scenario, subject, emotion_choices, emotion_label,
                                  cause_choices, cause_label}  → 2 questions per row

Output schema (jsonl):
  question_id, source=emobench, language=en|zh, task={EA|EU_emotion|EU_cause},
  story=scenario, question, options[4], gold (A-D)
"""
from __future__ import annotations
import json
from pathlib import Path

from datasets import load_dataset


QUESTION_EN = {
    "EU_emotion": "What is the most likely emotion of {subject}?",
    "EU_cause":   "What is the most likely cause of {subject}'s emotion?",
    "EA":         "What is the most appropriate response or action for {subject}?",
}
QUESTION_ZH = {
    "EU_emotion": "{subject}最可能的情绪是什么？",
    "EU_cause":   "{subject}的情绪最可能的原因是什么？",
    "EA":         "{subject}最合适的应对或行动是什么？",
}


def label_to_letter(choices: list, label: str) -> str | None:
    """EmoBench label is the FULL TEXT of the correct choice."""
    if label is None:
        return None
    label = label.strip()
    for i, c in enumerate(choices):
        if c == label or c.strip() == label:
            return chr(ord("A") + i)
    label_lc = label.lower()
    for i, c in enumerate(choices):
        if c.lower().strip() == label_lc:
            return chr(ord("A") + i)
    return None


def fmt_question(task: str, language: str, subject: str) -> str:
    tpl = (QUESTION_ZH if language == "zh" else QUESTION_EN)[task]
    s = subject if subject else ("主体" if language == "zh" else "the subject")
    return tpl.format(subject=s)


def emit(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    out = Path("data/eval/emobench_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    with open(out, "w") as f:
        # EA: 400 rows × choices/label
        ea = load_dataset("SahandSab/EmoBench", "emotional_application", split="train")
        print(f"=== EA: {len(ea)} rows, columns: {ea.column_names}")
        for i, row in enumerate(ea):
            lang = row["language"]
            choices = row.get("choices", [])
            label = row.get("label", "")
            if not isinstance(choices, list) or len(choices) < 2:
                n_skipped += 1; continue
            letter = label_to_letter(choices, label)
            if letter is None:
                n_skipped += 1; continue
            emit(f, {
                "question_id": f"emobench_EA_{lang}_{row['qid']}",
                "source": "emobench",
                "language": lang,
                "task": "EA",
                "subject": row.get("subject", ""),
                "category": row.get("category", ""),
                "question_type": row.get("question type", ""),
                "story": row["scenario"],
                "question": fmt_question("EA", lang, row.get("subject", "")),
                "options": choices,
                "gold": letter,
            })
            n_written += 1

        # EU: 400 rows × {emotion_choices/emotion_label, cause_choices/cause_label}
        eu = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train")
        print(f"=== EU: {len(eu)} rows, columns: {eu.column_names}")
        for i, row in enumerate(eu):
            lang = row["language"]
            subj = row.get("subject", "")
            scenario = row["scenario"]

            # Emit emotion question
            for sub_task, choices_key, label_key in [
                ("EU_emotion", "emotion_choices", "emotion_label"),
                ("EU_cause", "cause_choices", "cause_label"),
            ]:
                choices = row.get(choices_key, [])
                label = row.get(label_key, "")
                if not isinstance(choices, list) or len(choices) < 2:
                    n_skipped += 1; continue
                letter = label_to_letter(choices, label)
                if letter is None:
                    n_skipped += 1; continue
                emit(f, {
                    "question_id": f"emobench_{sub_task}_{lang}_{row['qid']}",
                    "source": "emobench",
                    "language": lang,
                    "task": sub_task,
                    "subject": subj,
                    "coarse_category": row.get("coarse_category", ""),
                    "finegrained_category": row.get("finegrained_category", ""),
                    "story": scenario,
                    "question": fmt_question(sub_task, lang, subj),
                    "options": choices,
                    "gold": letter,
                })
                n_written += 1

    print(f"\nWrote {n_written} records (skipped {n_skipped}) → {out}")


if __name__ == "__main__":
    main()

