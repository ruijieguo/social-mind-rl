"""Merge zh_translated.jsonl into tom_train.jsonl with eval-leakage filter."""
from __future__ import annotations
import json
import re
import random
import statistics as st
from collections import Counter
from pathlib import Path

import jsonlines
from datasketch import MinHash, MinHashLSH


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _4grams(text: str) -> set[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < 4:
        chars = [c for c in text.lower() if not c.isspace()]
        return {"".join(chars[i:i+4]) for i in range(len(chars) - 3)} if len(chars) >= 4 else set(chars)
    return {" ".join(toks[i:i+4]) for i in range(len(toks) - 3)}


def jaccard_4gram(a: str, b: str) -> float:
    A, B = _4grams(a), _4grams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _text_from_messages(rec: dict) -> str:
    user_msg = next((m for m in rec["messages"] if m["role"] == "user"), None)
    return user_msg["content"] if user_msg else ""


def _text_for_eval_record(rec: dict) -> str:
    parts = [rec.get("question"), rec.get("opt_a"), rec.get("opt_b"),
             rec.get("opt_c"), rec.get("opt_d")]
    return " ".join(str(p) for p in parts if p is not None)


def main():
    train_path = Path("data/tom/tom_train.jsonl")
    eval_path = Path("data/tom/tombench_eval.jsonl")
    zh_path = Path("data/tom/raw/zh_translated.jsonl")

    current_train = list(jsonlines.open(train_path))
    zh_new = list(jsonlines.open(zh_path))
    eval_records = list(jsonlines.open(eval_path))
    zh_eval = [r for r in eval_records if r.get("language") == "zh"]
    print(f"Loaded: train={len(current_train)}  zh_new={len(zh_new)}  zh_eval={len(zh_eval)}")

    print("indexing ToMBench zh eval ...")
    lsh = MinHashLSH(threshold=0.6, num_perm=128)
    eval_by_key = {}
    for r in zh_eval:
        mh = MinHash(num_perm=128)
        for g in _4grams(_text_for_eval_record(r)):
            mh.update(g.encode("utf8"))
        lsh.insert(r["question_id"], mh)
        eval_by_key[r["question_id"]] = _text_for_eval_record(r)

    kept = []
    dropped = 0
    max_jaccards = []
    for r in zh_new:
        text = _text_from_messages(r)
        mh = MinHash(num_perm=128)
        for g in _4grams(text):
            mh.update(g.encode("utf8"))
        candidates = lsh.query(mh)
        max_j = 0.0
        for c in candidates:
            j = jaccard_4gram(text, eval_by_key[c])
            if j > max_j:
                max_j = j
        max_jaccards.append(max_j)
        if max_j > 0.6:
            dropped += 1
        else:
            kept.append(r)

    print(f"Drop zh with leakage>0.6: kept={len(kept)} dropped={dropped}")
    if max_jaccards:
        sorted_j = sorted(max_jaccards, reverse=True)
        print(f"  top-5 Jaccard: {sorted_j[:5]}")
        print(f"  mean={st.mean(max_jaccards):.3f}")

    merged = current_train + kept
    with jsonlines.open(train_path, "w") as w:
        for r in merged:
            w.write(r)
    print(f"Wrote {len(merged)} records to {train_path}")

    rng = random.Random(42)
    subset = rng.sample(merged, k=min(4000, len(merged)))
    with jsonlines.open("data/tom/tom_train_4k.jsonl", "w") as w:
        for r in subset:
            w.write(r)
    print(f"Wrote {len(subset)} records to data/tom/tom_train_4k.jsonl")

    by_lang = Counter(r.get("language", "?") for r in merged)
    print(f"\nLanguage breakdown: {dict(by_lang)}")


if __name__ == "__main__":
    main()
