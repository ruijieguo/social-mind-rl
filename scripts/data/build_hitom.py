"""Build ~2k records from Hi-ToM via the ToM-RL author's generation scripts.

Strategy:
1. Look for prebuilt Hi-ToM parquet in HF (YangXiao-nlp/DynToM or bigai-ai)
2. Else, fall back to a smaller hand-curated set from the original Hi-ToM repo
"""
from __future__ import annotations
import json
import random
import subprocess
from pathlib import Path

import jsonlines

from scripts.data.schema import TomRecord


def _try_hf() -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("YangXiao-nlp/Hi-ToM", split="train", trust_remote_code=True)
        return list(ds)
    except Exception as e:
        print(f"HF Hi-ToM not available: {e}")
        return []


def _clone_and_generate() -> list[dict]:
    """Clone the ToM-RL repo (which bundles Hi-ToM generators) and run them."""
    work = Path("data/tom/raw/hi_tom_gen")
    work.mkdir(parents=True, exist_ok=True)
    repo = work / "ToM-RL"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/bigai-ai/ToM-RL", str(repo)],
            check=True,
        )
    # The Hi-ToM generator is at repo / scripts / hitom (path varies between commits)
    # Look for generated parquet directly first:
    parquet = repo / "data" / "cleaned_tom" / "ToM_train_HiEx_hint.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(parquet)
        return [
            {col: table[col][i].as_py() for col in table.column_names}
            for i in range(table.num_rows)
        ]
    print("Hi-ToM parquet not found in cloned repo; returning empty.")
    return []


def _row_to_record(row: dict, idx: int) -> TomRecord | None:
    """Coerce a Hi-ToM raw row into TomRecord."""
    story = row.get("story") or row.get("context") or row.get("prompt", "")
    question = row.get("question") or row.get("query", "")
    options = row.get("options") or row.get("choices")
    gold = row.get("answer") or row.get("label")
    if not (story and question and options and gold is not None):
        return None
    if isinstance(options, list) and len(options) >= 4:
        opts = list(options[:4])
    elif isinstance(options, dict) and {"A", "B", "C", "D"}.issubset(options):
        opts = [options["A"], options["B"], options["C"], options["D"]]
    else:
        return None
    gold_letter: str | None = None
    if isinstance(gold, int):
        gold_letter = "ABCD"[gold] if 0 <= gold < 4 else None
    elif isinstance(gold, str):
        g = gold.strip().upper()
        if g in {"A", "B", "C", "D"}:
            gold_letter = g
        else:
            for j, o in enumerate(opts):
                if str(o).strip() == gold.strip():
                    gold_letter = "ABCD"[j]
                    break
    if gold_letter is None:
        return None
    return TomRecord(
        question_id=f"hitom_{idx}",
        source="hi_tom", language="en", task="False Belief",
        story=story, question=question,
        opt_a=str(opts[0]), opt_b=str(opts[1]),
        opt_c=str(opts[2]), opt_d=str(opts[3]),
        gold=gold_letter,
    )


def main():
    random.seed(42)
    out = Path("data/tom/raw/hi_tom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _try_hf() or _clone_and_generate()
    if not rows:
        print("Hi-ToM unavailable; writing empty file (other sources will compensate)")
        with jsonlines.open(out, "w") as w:
            pass
        return

    records: list[TomRecord] = []
    random.shuffle(rows)
    for i, row in enumerate(rows[:5000]):
        if len(records) >= 2000:
            break
        rec = _row_to_record(row, i)
        if rec is not None:
            records.append(rec)

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
