"""Re-shuffle synth options to balance the gold-letter distribution.

The deepseek-v4-flash synthesizer consistently puts the correct answer in
options A/B (~89%), leaving D nearly empty. A model trained on this would
learn the shortcut "guess A or B" instead of actual reasoning.

Fix: for each synth record, randomly permute the 4 options and recompute
the gold letter. This preserves story/question/correctness; only the
option ordering changes.
"""
from __future__ import annotations
import json
import random
import re
from collections import Counter
from pathlib import Path

import jsonlines


# Match: "故事：" or "Story:" then content then "A. ..." through "D. ..."
_OPTS_RE = re.compile(
    r"(?P<head>.*?)\n"
    r"(?P<q>(?:Question|问题).*?)\n"
    r"A\.\s*(?P<a>.+?)\n"
    r"B\.\s*(?P<b>.+?)\n"
    r"C\.\s*(?P<c>.+?)\n"
    r"D\.\s*(?P<d>.+?)$",
    re.DOTALL,
)


def reshuffle_record(rec: dict, rng: random.Random) -> dict:
    """Return a new record with options re-shuffled and gold letter updated."""
    user_msg = next((m for m in rec["messages"] if m["role"] == "user"), None)
    if not user_msg:
        return rec

    m = _OPTS_RE.match(user_msg["content"])
    if not m:
        return rec

    opts = [m.group("a").strip(), m.group("b").strip(),
            m.group("c").strip(), m.group("d").strip()]
    gold_idx = "ABCD".index(rec["ground_truth"])

    order = list(range(4))
    rng.shuffle(order)
    new_opts = [opts[j] for j in order]
    new_gold_idx = order.index(gold_idx)
    new_gold = "ABCD"[new_gold_idx]

    head = m.group("head")
    q = m.group("q").strip()
    new_user = (
        f"{head}\n{q}\n"
        f"A. {new_opts[0]}\n"
        f"B. {new_opts[1]}\n"
        f"C. {new_opts[2]}\n"
        f"D. {new_opts[3]}"
    )

    new_rec = dict(rec)
    new_rec["ground_truth"] = new_gold
    new_rec["messages"] = [
        m if m["role"] != "user" else {"role": "user", "content": new_user}
        for m in rec["messages"]
    ]
    return new_rec


def main():
    inp = Path("data/tom/tom_train.jsonl")
    out = Path("data/tom/tom_train.jsonl")  # in-place rewrite

    rng = random.Random(42)
    records = list(jsonlines.open(inp))
    before = Counter(r["ground_truth"] for r in records if r["source"] == "synth")

    out_records = []
    n_reshuffled = 0
    for r in records:
        if r["source"] == "synth":
            new_r = reshuffle_record(r, rng)
            if new_r["ground_truth"] != r["ground_truth"] or new_r["messages"] != r["messages"]:
                n_reshuffled += 1
            out_records.append(new_r)
        else:
            out_records.append(r)

    after = Counter(r["ground_truth"] for r in out_records if r["source"] == "synth")

    with jsonlines.open(out, "w") as w:
        for r in out_records:
            w.write(r)

    n_synth = sum(before.values())
    print(f"Reshuffled {n_reshuffled}/{n_synth} synth records.")
    print(f"\nBefore: {dict(before)}")
    print(f"After:  {dict(after)}")
    for g in "ABCD":
        b_pct = 100 * before.get(g, 0) / n_synth
        a_pct = 100 * after.get(g, 0) / n_synth
        print(f"  {g}: {b_pct:5.1f}%  →  {a_pct:5.1f}%")

    # Also regenerate the 4k subset to inherit the rebalanced synth records
    rng2 = random.Random(42)
    subset = rng2.sample(out_records, k=min(4000, len(out_records)))
    with jsonlines.open("data/tom/tom_train_4k.jsonl", "w") as w:
        for r in subset:
            w.write(r)
    print(f"\nAlso rewrote data/tom/tom_train_4k.jsonl ({len(subset)} records).")


if __name__ == "__main__":
    main()
