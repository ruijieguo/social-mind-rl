"""Build ~2k records from ExploreToM (facebookresearch/ExploreToM)."""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def main():
    random.seed(42)
    out = Path("data/tom/raw/exploretom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    candidate_repos = [
        ("facebook/ExploreToM", None),
        ("facebookresearch/ExploreToM", None),
    ]
    ds = None
    for repo, split in candidate_repos:
        try:
            ds = load_dataset(repo, split=split or "train", trust_remote_code=True)
            print(f"loaded {repo}")
            break
        except Exception as e:
            print(f"could not load {repo}: {e}")
    if ds is None:
        print("ExploreToM not available; writing empty file (will continue without this source)")
        with jsonlines.open(out, "w") as w:
            pass
        return

    n_target = 2000
    records: list[TomRecord] = []
    for i, row in enumerate(ds.shuffle(seed=42).select(range(min(len(ds), n_target * 2)))):
        if len(records) >= n_target:
            break
        story = row.get("story") or row.get("context", "")
        question = row.get("question", "")
        # Some ExploreToM splits provide options; if not, skip
        opts = row.get("options")
        gold = row.get("answer")
        if not (story and question and opts and gold):
            continue
        if isinstance(opts, list) and len(opts) >= 4:
            opt_list = list(opts[:4])
        else:
            continue
        # gold may be index, letter, or text
        gold_letter: str | None = None
        if isinstance(gold, int) and 0 <= gold < 4:
            gold_letter = "ABCD"[gold]
        elif isinstance(gold, str):
            if gold.strip().upper() in {"A", "B", "C", "D"}:
                gold_letter = gold.strip().upper()
            else:
                for j, o in enumerate(opt_list):
                    if str(o).strip() == gold.strip():
                        gold_letter = "ABCD"[j]
                        break
        if gold_letter is None:
            continue
        records.append(TomRecord(
            question_id=f"exploretom_{i}",
            source="exploretom", language="en", task="False Belief",
            story=story, question=question,
            opt_a=str(opt_list[0]), opt_b=str(opt_list[1]),
            opt_c=str(opt_list[2]), opt_d=str(opt_list[3]),
            gold=gold_letter,
        ))

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
