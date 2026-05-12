"""Build ~1k records from SimpleToM (allenai/simpletom)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def _transform_to_mcq(row: dict, idx: int) -> TomRecord | None:
    """SimpleToM has yes/no behavior-prediction; convert to 4-MCQ with synonyms."""
    story = row.get("story") or row.get("context") or ""
    question = row.get("question") or row.get("mental_state_question") or ""
    gold_bool = row.get("answer") or row.get("label")  # boolean or "yes"/"no"
    if isinstance(gold_bool, bool):
        gold_yes = gold_bool
    elif isinstance(gold_bool, str):
        gold_yes = gold_bool.lower().startswith("y")
    else:
        return None
    # 4 options for a yes/no flavor
    opts = ["Yes, they will", "No, they will not", "Cannot be determined", "Both yes and no"]
    gold_letter = "A" if gold_yes else "B"
    if not story or not question:
        return None
    return TomRecord(
        question_id=f"simpletom_{idx}",
        source="simpletom", language="en", task="Other",
        story=story, question=question,
        opt_a=opts[0], opt_b=opts[1], opt_c=opts[2], opt_d=opts[3],
        gold=gold_letter,
    )


def main():
    random.seed(42)
    out = Path("data/tom/raw/simpletom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("allenai/SimpleToM", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"SimpleToM not available via HF datasets: {e}")
        print("Skipping SimpleToM source; merge_and_dedupe will fall back to other sources.")
        with jsonlines.open(out, "w") as w:
            pass
        return

    records: list[TomRecord] = []
    for i, row in enumerate(ds.shuffle(seed=42).select(range(min(1500, len(ds))))):
        if len(records) >= 1000:
            break
        rec = _transform_to_mcq(row, i)
        if rec is not None:
            records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
