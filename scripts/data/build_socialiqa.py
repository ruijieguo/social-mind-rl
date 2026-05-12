"""Build ~1.5k records from SocialIQa (allenai/social_i_qa)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def main():
    random.seed(42)
    ds = load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
    # ds fields: context, question, answerA, answerB, answerC, label ('1'/'2'/'3')

    out = Path("data/tom/raw/socialiqa.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert 3-option to 4-option by adding "None of the above" as distractor
    # then shuffle to randomize gold position
    records: list[TomRecord] = []
    n = 1500
    for i, row in enumerate(ds.shuffle(seed=42).select(range(n * 2))):
        # Try to get 1500 well-formed; some may be skipped
        if len(records) >= n:
            break
        opts = [row["answerA"], row["answerB"], row["answerC"], "None of the above"]
        label = row["label"].strip()
        if label not in {"1", "2", "3"}:
            continue
        gold_idx = int(label) - 1
        # Shuffle 4 options
        idxs = list(range(4))
        random.shuffle(idxs)
        new_opts = [opts[j] for j in idxs]
        new_gold = "ABCD"[idxs.index(gold_idx)]
        rec = TomRecord(
            question_id=f"socialiqa_{i}",
            source="socialiqa", language="en", task="Other",
            story=row["context"], question=row["question"],
            opt_a=new_opts[0], opt_b=new_opts[1],
            opt_c=new_opts[2], opt_d=new_opts[3],
            gold=new_gold,
        )
        records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
