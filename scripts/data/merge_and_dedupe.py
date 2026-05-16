"""Merge all data sources, dedupe internally (MinHash), and cross-check vs ToMBench eval.

Outputs:
- data/tom/tom_train.jsonl              (~8k training records)
- data/tom/tom_train_4k.jsonl           (random 4k subset for stage-1)
- data/tom/dedup_report.json            (audit: max-Jaccard distribution)
- docs/data-card.md                     (auto-generated)
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Iterable

import jsonlines
from datasketch import MinHash, MinHashLSH

from scripts.data.schema import TomRecord


# 4-gram tokenizer
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _4grams(text: str) -> set[str]:
    """Return set of word 4-grams for Jaccard / MinHash.

    For texts with fewer than 5 tokens, use 3-grams to ensure overlap detection.
    For longer texts, use 4-grams.
    """
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < 3:
        # Very short: use individual tokens
        return set(toks) if toks else set()
    elif len(toks) < 5:
        # Short: use 3-grams to get better overlap detection
        return {" ".join(toks[i:i+3]) for i in range(len(toks) - 2)}
    else:
        # Long enough: use 4-grams
        return {" ".join(toks[i:i+4]) for i in range(len(toks) - 3)}


def jaccard_4gram(a: str, b: str) -> float:
    A, B = _4grams(a), _4grams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


class MinHashIndex:
    def __init__(self, threshold: float = 0.5, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.docs: dict[str, set[str]] = {}
        self.num_perm = num_perm

    def _mh(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        for g in _4grams(text):
            mh.update(g.encode("utf8"))
        return mh

    def add(self, key: str, text: str) -> None:
        self.docs[key] = _4grams(text)
        self.lsh.insert(key, self._mh(text))

    def query(self, exclude_key: str, text: str) -> list[str]:
        cands = self.lsh.query(self._mh(text))
        return [c for c in cands if c != exclude_key]


def build_minhash_index(corpus: Iterable[tuple[str, str]], threshold: float = 0.5) -> MinHashIndex:
    idx = MinHashIndex(threshold=threshold)
    for key, text in corpus:
        idx.add(key, text)
    return idx


def _text_for_match(rec: dict) -> str:
    """Canonical text used for similarity: question + 4 options."""
    parts = [rec.get("question"), rec.get("opt_a"), rec.get("opt_b"), rec.get("opt_c"), rec.get("opt_d")]
    return " ".join(str(p) for p in parts if p is not None)


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with jsonlines.open(path) as r:
        return list(r)


def main():
    random.seed(42)
    raw_dir = Path("data/tom/raw")
    eval_path = Path("data/tom/tombench_eval.jsonl")
    out_full = Path("data/tom/tom_train.jsonl")
    out_4k = Path("data/tom/tom_train_4k.jsonl")
    report_path = Path("data/tom/dedup_report.json")

    sources = {
        "hi_tom":     raw_dir / "hi_tom.jsonl",
        "exploretom": raw_dir / "exploretom.jsonl",
        "simpletom":  raw_dir / "simpletom.jsonl",
        "socialiqa":  raw_dir / "socialiqa.jsonl",
        "synth":      raw_dir / "synth.jsonl",
        # Phase 1 (2026-05-16) targeted synthesis. See docs/badcase_analysis.md.
        # Each source already has source="synth_phase1" in its records, so they
        # remain identifiable in the merged file.
        "synth_phase1_faux_pas":  raw_dir / "synth_phase1_faux_pas.jsonl",
        "synth_phase1_hinting":   raw_dir / "synth_phase1_hinting.jsonl",
        "synth_phase1_scalar":    raw_dir / "synth_phase1_scalar.jsonl",
        "synth_phase1_so_belief": raw_dir / "synth_phase1_so_belief.jsonl",
    }

    # Step 1: Load ToMBench eval into MinHash index
    print("indexing ToMBench eval ...")
    eval_records = _load_raw(eval_path)
    eval_index = MinHashIndex(threshold=0.6)
    for r in eval_records:
        eval_index.add(r["question_id"], _text_for_match(r))

    # Step 2: For each source, drop any train record similar to any eval record
    max_jaccard_by_source: dict[str, list[float]] = {k: [] for k in sources}
    survivors: list[dict] = []
    dropped_by_leakage = 0
    for src, path in sources.items():
        rows = _load_raw(path)
        print(f"  {src}: loaded {len(rows)} rows from {path}")
        for r in rows:
            text = _text_for_match(r)
            cand = eval_index.query(exclude_key="", text=text)
            max_j = 0.0
            for c in cand:
                # Compute exact Jaccard for candidate
                j = jaccard_4gram(text, _text_for_match(next(e for e in eval_records if e["question_id"] == c)))
                if j > max_j:
                    max_j = j
            max_jaccard_by_source[src].append(max_j)
            if max_j > 0.6:
                dropped_by_leakage += 1
                continue
            survivors.append(r)

    print(f"  total after eval-leakage filter: {len(survivors)} (dropped {dropped_by_leakage})")

    # Step 3: Internal dedupe among survivors
    print("internal dedupe ...")
    internal_index = MinHashIndex(threshold=0.7)
    seen: list[dict] = []
    dropped_internal = 0
    for r in survivors:
        text = _text_for_match(r)
        dups = internal_index.query(exclude_key=r["question_id"], text=text)
        # Confirm with exact Jaccard
        is_dup = False
        for d in dups:
            other = next(s for s in seen if s["question_id"] == d)
            if jaccard_4gram(text, _text_for_match(other)) > 0.7:
                is_dup = True
                break
        if is_dup:
            dropped_internal += 1
            continue
        internal_index.add(r["question_id"], text)
        seen.append(r)

    print(f"  total after internal dedupe: {len(seen)} (dropped {dropped_internal})")

    # Step 4: Build messages field for ROLL training format
    from scripts.eval.run_tombench import SYSTEM_PROMPT_DIRECT, build_user_prompt_en, build_user_prompt_zh

    train_records: list[dict] = []
    for r in seen:
        builder = build_user_prompt_zh if r["language"] == "zh" else build_user_prompt_en
        user_text = builder(
            story=r["story"], question=r["question"],
            opt_a=r["opt_a"], opt_b=r["opt_b"],
            opt_c=r["opt_c"], opt_d=r["opt_d"],
        )
        train_records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
                {"role": "user", "content": user_text},
            ],
            "ground_truth": r["gold"],
            "tag": "tom_mcq",
            "source": r["source"],
            "language": r["language"],
            "task": r["task"],
            "question_id": r["question_id"],
        })

    out_full.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_full, "w") as w:
        for r in train_records:
            w.write(r)
    print(f"wrote {len(train_records)} train records to {out_full}")

    # Step 5: Fold in pre-translated zh records (training-format already).
    # These were produced by translate_to_zh.py and have already gone through
    # the EN training pool, so leakage was indirectly controlled. Still
    # re-check zh-vs-zh-eval similarity to be safe.
    zh_path = raw_dir / "zh_translated.jsonl"
    zh_kept = 0
    zh_dropped = 0
    if zh_path.exists():
        print(f"merging zh translations from {zh_path} ...")
        zh_records = _load_raw(zh_path)
        # Index zh eval rows
        zh_eval = [r for r in eval_records if r.get("language") == "zh"]
        zh_eval_index = MinHashIndex(threshold=0.6)
        for er in zh_eval:
            zh_eval_index.add(er["question_id"], _text_for_match(er))

        with jsonlines.open(out_full, "a") as w:
            for zr in zh_records:
                user_msg = next((m for m in zr.get("messages", []) if m["role"] == "user"), None)
                zh_text = user_msg["content"] if user_msg else ""
                cands = zh_eval_index.query(exclude_key="", text=zh_text)
                max_j = 0.0
                for c in cands:
                    er = next(e for e in zh_eval if e["question_id"] == c)
                    j = jaccard_4gram(zh_text, _text_for_match(er))
                    if j > max_j:
                        max_j = j
                if max_j > 0.6:
                    zh_dropped += 1
                else:
                    w.write(zr)
                    zh_kept += 1
        print(f"  zh: kept={zh_kept} dropped={zh_dropped}")

    # Re-count after appending
    all_after_zh = list(jsonlines.open(out_full))
    print(f"final tom_train.jsonl size: {len(all_after_zh)} records")

    # Subset
    subset = random.sample(all_after_zh, k=min(4000, len(all_after_zh)))
    with jsonlines.open(out_4k, "w") as w:
        for r in subset:
            w.write(r)
    print(f"wrote {len(subset)} subset records to {out_4k}")

    # Dedup report
    report = {
        "n_total_survived": len(train_records),
        "n_dropped_by_eval_leakage": dropped_by_leakage,
        "n_dropped_by_internal_dedupe": dropped_internal,
        "per_source_max_jaccard_distribution": {
            src: {
                "mean": sum(v) / len(v) if v else 0.0,
                "max": max(v) if v else 0.0,
                "p95": sorted(v)[int(0.95 * len(v))] if v else 0.0,
                "n": len(v),
            }
            for src, v in max_jaccard_by_source.items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote dedup report to {report_path}")

    # Auto-generated data card
    card = Path("docs/data-card.md")
    lines = ["# Data Card", "", f"Generated from `merge_and_dedupe.py`.", ""]
    lines.append("## Sources after dedupe")
    by_src: dict[str, int] = {}
    for r in train_records:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    for src, n in sorted(by_src.items()):
        lines.append(f"- **{src}**: {n}")
    lines.append("")
    lines.append("## Leakage audit")
    lines.append(f"- Records dropped (>0.6 Jaccard with any ToMBench eval question): {dropped_by_leakage}")
    lines.append(f"- Records dropped (internal near-dup >0.7): {dropped_internal}")
    lines.append("")
    lines.append("## Per-source max-Jaccard vs ToMBench eval")
    lines.append("| Source | n | mean | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for src, st in report["per_source_max_jaccard_distribution"].items():
        lines.append(f"| {src} | {st['n']} | {st['mean']:.3f} | {st['p95']:.3f} | {st['max']:.3f} |")
    card.write_text("\n".join(lines))
    print(f"wrote data card to {card}")


if __name__ == "__main__":
    main()
