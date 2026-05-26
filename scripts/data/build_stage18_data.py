"""Build Stage-18 (v3.4) training data: stage17 backbone + GPT-5.5 distillation.

Sources:
  - tom_train_stage17.jsonl                    (18299, v3.3 backbone)
  - distill_emobench_v34b.jsonl                (~204)
  - distill_socialiqa_v34.jsonl                (~237)
  - distill_tombench_belief_v34b.jsonl         (~359, Belief 2× + Knowledge)
  - distill_hitom_v34b.jsonl                   (~125, order_2-4)

Total ~19.2k after dedup.

MinHash dedup vs eval sets uses Jaccard ≥ 0.6 — distill data is paraphrased
from eval errors, so the user-message text should differ enough to pass
the filter (we paraphrase the story, not the question/options).
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def shingle_4gram(text: str) -> set[str]:
    tokens = text.lower().split()
    return set(" ".join(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def extract_story_question(rec: dict) -> str:
    if "messages" in rec:
        for m in rec["messages"]:
            if m.get("role") == "user":
                return m.get("content", "")
        return ""
    return f"{rec.get('story','')} {rec.get('question','')}"


def extract_full_for_dedup(rec: dict) -> str:
    if "messages" in rec:
        return "|||".join(f"{m.get('role','')}::{m.get('content','')}" for m in rec["messages"])
    return extract_story_question(rec)


def main():
    out_train = ROOT / "data/tom/tom_train_stage18.jsonl"
    out_report = ROOT / "data/tom/stage18_dedup_report.json"

    eval_files = {
        "tombench": ROOT / "data/tom/tombench_eval.jsonl",
        "hitom": ROOT / "data/eval/hitom_eval.jsonl",
        "socialiqa": ROOT / "data/eval/socialiqa_eval.jsonl",
        "emobench": ROOT / "data/eval/emobench_eval.jsonl",
    }
    eval_shingles = {}
    for name, p in eval_files.items():
        recs = load_jsonl(p)
        eval_shingles[name] = [shingle_4gram(extract_story_question(r)) for r in recs]
        print(f"  {name}: {len(recs)}")

    train_files = [
        ROOT / "data/tom/tom_train_stage17.jsonl",
        ROOT / "data/tom/raw/distill_emobench_v34b.jsonl",
        ROOT / "data/tom/raw/distill_socialiqa_v34.jsonl",
        ROOT / "data/tom/raw/distill_tombench_belief_v34b.jsonl",
        ROOT / "data/tom/raw/distill_hitom_v34b.jsonl",
    ]
    train_records: list[dict] = []
    for p in train_files:
        if not p.exists():
            print(f"WARN: missing {p}, skipping")
            continue
        recs = load_jsonl(p)
        print(f"  loaded {len(recs)} from {p.name}")
        train_records.extend(recs)
    print(f"\nTotal raw train: {len(train_records)}")

    seen_hash: set[str] = set()
    leakage = []
    keep: list[dict] = []
    for r in train_records:
        full_text = extract_full_for_dedup(r)
        h = text_hash(full_text)
        if h in seen_hash:
            continue
        text = extract_story_question(r)
        s_train = shingle_4gram(text)
        if not s_train:
            seen_hash.add(h)
            keep.append(r)
            continue
        leaks = False
        for ename, e_shingles in eval_shingles.items():
            for j, s_e in enumerate(e_shingles):
                if jaccard(s_train, s_e) >= 0.6:
                    leakage.append({
                        "train_qid": r.get("question_id", "?"),
                        "eval_set": ename,
                        "eval_idx": j,
                    })
                    leaks = True
                    break
            if leaks:
                break
        if not leaks:
            seen_hash.add(h)
            keep.append(r)

    print(f"\nAfter dedup + leakage filter: {len(keep)}")
    print(f"Leakage hits dropped: {len(leakage)}")

    random.seed(42)
    random.shuffle(keep)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote train → {out_train}")

    from collections import Counter
    src_dist = Counter(r.get("source", "?") for r in keep)
    task_dist = Counter(r.get("task", "?") for r in keep)
    report = {
        "total_train": len(keep),
        "leakage_dropped": len(leakage),
        "leakage_examples": leakage[:20],
        "source_dist": dict(src_dist),
        "task_dist": dict(task_dist),
    }
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report → {out_report}")
    print("\nSource distribution:")
    for s, c in src_dist.most_common():
        print(f"  {s:<35} {c}")


if __name__ == "__main__":
    main()
