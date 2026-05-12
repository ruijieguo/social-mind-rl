"""Build ~2k MCQ records from ExploreToM (facebook/ExploreToM).

ExploreToM ships as open-ended QA: each row has `story_structure`,
`infilled_story`, `question`, `expected_answer` (free text). To turn it
into a 4-option MCQ we:

1. Use `infilled_story` as the story (narrative prose, more realistic than
   the structural template).
2. The gold answer is `expected_answer` itself.
3. Build 3 distractors by sampling other rows' `expected_answer`s,
   filtered to ensure they differ from the gold by surface string.
4. Shuffle the four options and record the gold letter.

We restrict to false-belief rows (sprop=is_false_belief_story_1st or
is_false_belief_story_1st_and_2nd) since those probe real ToM, and we
prefer non_unique_mental_state rows to avoid trivial questions.
"""
from __future__ import annotations
import random
from pathlib import Path

import jsonlines
from datasets import load_dataset

from scripts.data.schema import TomRecord


def _sample_distractors(rng: random.Random, gold: str, pool: list[str], n: int = 3) -> list[str]:
    """Sample n distractors from the answer pool, different from gold."""
    tries = 0
    out: list[str] = []
    while len(out) < n and tries < 50:
        cand = rng.choice(pool)
        if cand != gold and cand not in out:
            out.append(cand)
        tries += 1
    while len(out) < n:
        # Fallback synthetic distractor in case the pool is too thin.
        out.append(f"(distractor option {len(out)+1})")
    return out


def main():
    rng = random.Random(42)
    out = Path("data/tom/raw/exploretom.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("facebook/ExploreToM", split="train")
    except Exception as e:
        print(f"ExploreToM unavailable: {e}")
        with jsonlines.open(out, "w") as w:
            pass
        return

    # Build the answer pool from all rows for use as distractors.
    answer_pool = [row["expected_answer"] for row in ds if row.get("expected_answer")]
    answer_pool = list({a for a in answer_pool if isinstance(a, str) and a.strip()})
    print(f"loaded {len(ds)} rows; distractor pool size {len(answer_pool)}")

    # Prefer false-belief rows (per ToM-RL paper: they probe real ToM ability).
    fb_indices = [
        i for i, row in enumerate(ds)
        if row.get("sprop=is_false_belief_story_1st") or row.get("sprop=is_false_belief_story_1st_and_2nd")
    ]
    print(f"  false-belief rows: {len(fb_indices)}")
    # Shuffle and cap at 2000.
    rng.shuffle(fb_indices)
    selected = fb_indices[:2000] if len(fb_indices) >= 2000 else fb_indices
    print(f"  selected: {len(selected)}")

    records: list[TomRecord] = []
    for idx in selected:
        row = ds[idx]
        story = row.get("infilled_story") or row.get("story_structure") or ""
        question = row.get("question", "")
        gold = row.get("expected_answer", "")
        if not (story and question and gold and isinstance(gold, str)):
            continue
        distractors = _sample_distractors(rng, gold, answer_pool, n=3)
        opts4 = [gold] + distractors
        order = list(range(4))
        rng.shuffle(order)
        new_opts = [opts4[j] for j in order]
        new_gold = "ABCD"[order.index(0)]
        records.append(TomRecord(
            question_id=f"exploretom_{idx}",
            source="exploretom", language="en", task="False Belief",
            story=story, question=question,
            opt_a=new_opts[0], opt_b=new_opts[1],
            opt_c=new_opts[2], opt_d=new_opts[3],
            gold=new_gold,
        ))

    with jsonlines.open(out, "w") as w:
        for r in records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
