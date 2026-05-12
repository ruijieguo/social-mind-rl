"""Build ~1k records from SimpleToM (allenai/SimpleToM)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


# Per-config canonical distractor pool, used to expand 2-option (A/B) into
# a 4-option MCQ. Original options remain at indices 0,1; distractors fill 2,3
# then the four options are shuffled and the gold letter is recomputed.
_DISTRACTOR_POOLS = {
    "mental-state-qa":  ["Cannot be determined from the story", "Both yes and no"],
    "behavior-qa":      ["Refuse to act and stand still", "Ask someone else for help"],
    "judgment-qa":      ["Mostly reasonable but with minor concerns", "Cannot be judged from the story"],
}


def _transform_to_mcq(row: dict, idx: int, distractors: list[str], task_tag: str) -> TomRecord | None:
    story = row.get("story", "")
    question = row.get("question", "")
    choices = row.get("choices") or {}
    texts = choices.get("text") if isinstance(choices, dict) else None
    labels = choices.get("label") if isinstance(choices, dict) else None
    gold_label = row.get("answerKey", "")
    if not (story and question and texts and labels and gold_label in {"A", "B"}):
        return None
    if len(texts) < 2:
        return None

    # Build a 4-option pool: original two + 2 distractors
    opts4 = [texts[0], texts[1], distractors[0], distractors[1]]
    gold_idx = labels.index(gold_label)  # 0 or 1
    rng = random.Random(idx)
    order = list(range(4))
    rng.shuffle(order)
    new_opts = [opts4[j] for j in order]
    new_gold = "ABCD"[order.index(gold_idx)]

    return TomRecord(
        question_id=f"simpletom_{task_tag}_{idx}",
        source="simpletom", language="en", task="Other",
        story=story, question=question,
        opt_a=new_opts[0], opt_b=new_opts[1],
        opt_c=new_opts[2], opt_d=new_opts[3],
        gold=new_gold,
    )


def main():
    random.seed(42)
    out = Path("data/tom/raw/simpletom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[TomRecord] = []
    # Load all three QA configs; story-data is descriptive metadata, skip it.
    for cfg in ["mental-state-qa", "behavior-qa", "judgment-qa"]:
        try:
            ds = load_dataset("allenai/SimpleToM", cfg, split="test")
        except Exception as e:
            print(f"SimpleToM cfg={cfg} unavailable: {e}")
            continue
        distractors = _DISTRACTOR_POOLS[cfg]
        print(f"loaded {cfg}: {len(ds)} rows")
        for i, row in enumerate(ds):
            rec = _transform_to_mcq(row, i, distractors, cfg)
            if rec is not None:
                records.append(rec)

    # Cap at ~1000 to match plan
    if len(records) > 1000:
        random.Random(42).shuffle(records)
        records = records[:1000]

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
