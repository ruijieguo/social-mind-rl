"""Download ToMBench from GitHub and convert to unified JSONL.

Outputs:
- data/tom/tombench_eval.jsonl  (one record per (question, language))
- data/tom/tombench_eval_subset500.jsonl  (random 500 for training-time eval)
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import jsonlines

from scripts.data.schema import TomRecord, ability_to_task


TOMBENCH_GITHUB_BASE = "https://raw.githubusercontent.com/zhchen18/ToMBench/main/data"
TOMBENCH_FILES = [
    "Ambiguous Story Task.jsonl",
    "Completion of Failed Actions.jsonl",
    "Discrepant Desires.jsonl",
    "Discrepant Emotions.jsonl",
    "Discrepant Intentions.jsonl",
    "Emotion Regulation.jsonl",
    "False Belief Task.jsonl",
    "Faux-pas Recognition Test.jsonl",
    "Hidden Emotions.jsonl",
    "Hinting Task Test.jsonl",
    "Knowledge-Attention Links.jsonl",
    "Knowledge-Pretend Play Links.jsonl",
    "Moral Emotions.jsonl",
    "Multiple Desires.jsonl",
    "Percepts-Knowledge Links.jsonl",
    "Persuasion Story Task.jsonl",
    "Prediction of Actions.jsonl",
    "Scalar Implicature Test.jsonl",
    "Strange Story Task.jsonl",
    "Unexpected Outcome Test.jsonl",
]


def download_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname in TOMBENCH_FILES:
        local = out_dir / fname
        if local.exists():
            paths.append(local)
            continue
        url = f"{TOMBENCH_GITHUB_BASE}/{fname.replace(' ', '%20')}"
        print(f"downloading {fname} ...")
        urlretrieve(url, local)
        paths.append(local)
    return paths


def _str_field(raw: dict, *keys: str) -> str:
    """Get a field as str, coalescing missing/NaN/non-string values to ''.

    ToMBench source data uses NaN (the float, not the JSON null) for absent
    options on 3-option questions like Faux-pas Recognition Test. Plain
    `raw.get(k) or ''` returns NaN because float('nan') is truthy.
    """
    for k in keys:
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            # NaN check via != self (only NaN is unequal to itself)
            if v != v:
                continue
            return str(v)
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def transform_one(raw: dict, idx_in_file: int, fname: str) -> list[TomRecord]:
    """One raw ToMBench entry → 2 records (en + zh)."""
    ability = raw.get("能力\nABILITY", "")
    task = ability_to_task(ability)
    qid_base = f"{fname.replace('.jsonl','').replace(' ', '_')}_{idx_in_file}"
    gold = _str_field(raw, "答案\nANSWER", "ANSWER")
    if gold not in {"A", "B", "C", "D"}:
        return []

    records = []
    # English
    en_story = _str_field(raw, "STORY")
    en_q = _str_field(raw, "QUESTION")
    en_a = _str_field(raw, "OPTION-A")
    en_b = _str_field(raw, "OPTION-B")
    en_c = _str_field(raw, "OPTION-C")
    en_d = _str_field(raw, "OPTION-D")
    if en_story and en_q and en_a:
        records.append(TomRecord(
            question_id=f"{qid_base}_en", source="tombench",
            language="en", task=task,
            story=en_story, question=en_q,
            opt_a=en_a, opt_b=en_b, opt_c=en_c, opt_d=en_d,
            gold=gold,
        ))
    # Chinese
    zh_story = _str_field(raw, "故事")
    zh_q = _str_field(raw, "问题")
    zh_a = _str_field(raw, "选项A")
    zh_b = _str_field(raw, "选项B")
    zh_c = _str_field(raw, "选项C")
    zh_d = _str_field(raw, "选项D")
    if zh_story and zh_q and zh_a:
        records.append(TomRecord(
            question_id=f"{qid_base}_zh", source="tombench",
            language="zh", task=task,
            story=zh_story, question=zh_q,
            opt_a=zh_a, opt_b=zh_b, opt_c=zh_c, opt_d=zh_d,
            gold=gold,
        ))
    return records


def main():
    raw_dir = Path("data/tom/raw/tombench")
    out_full = Path("data/tom/tombench_eval.jsonl")
    out_sub = Path("data/tom/tombench_eval_subset500.jsonl")

    paths = download_all(raw_dir)
    all_records: list[TomRecord] = []
    for p in paths:
        with jsonlines.open(p) as reader:
            for idx, raw in enumerate(reader):
                all_records.extend(transform_one(raw, idx, p.name))

    out_full.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_full, "w") as w:
        for r in all_records:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(all_records)} records to {out_full}")

    random.seed(42)
    subset = random.sample(all_records, k=min(500, len(all_records)))
    with jsonlines.open(out_sub, "w") as w:
        for r in subset:
            w.write(r.to_jsonl_dict())
    print(f"wrote {len(subset)} subset records to {out_sub}")


if __name__ == "__main__":
    main()
