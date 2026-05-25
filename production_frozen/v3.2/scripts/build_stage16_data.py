"""Build Stage-16 training data + validation slices.

Step 1: Merge all synth + backbone into tom_train_stage16.jsonl
Step 2: MinHash dedup vs ALL 4 eval sets (tombench_eval, hitom_eval,
        socialiqa_eval, emobench_eval)
Step 3: Carve out validation slices (200 hitom + 100 EU_emotion) from
        the eval sets, NOT touching the train data.

Output:
  data/tom/tom_train_stage16.jsonl              # training corpus
  data/tom/hitom_eval_val200.jsonl              # validation slice
  data/tom/emobench_eu_emotion_val100.jsonl     # validation slice
  data/tom/stage16_dedup_report.json
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
    """Get story+question text from a training record (messages format)."""
    if "messages" in rec:
        # User message contains story + question
        for m in rec["messages"]:
            if m.get("role") == "user":
                return m.get("content", "")
        return ""
    # Eval format
    return f"{rec.get('story','')} {rec.get('question','')}"


def main():
    out_train = ROOT / "data/tom/tom_train_stage16.jsonl"
    out_val_hitom = ROOT / "data/tom/hitom_eval_val200.jsonl"
    out_val_emobench = ROOT / "data/tom/emobench_eu_emotion_val100.jsonl"
    out_report = ROOT / "data/tom/stage16_dedup_report.json"

    # 1) Load eval sets for leakage check
    eval_files = {
        "tombench": ROOT / "data/tom/tombench_eval.jsonl",
        "hitom": ROOT / "data/eval/hitom_eval.jsonl",
        "socialiqa": ROOT / "data/eval/socialiqa_eval.jsonl",
        "emobench": ROOT / "data/eval/emobench_eval.jsonl",
    }
    print("== Eval set sizes ==")
    eval_shingles: dict[str, list[set[str]]] = {}
    eval_records: dict[str, list[dict]] = {}
    for name, p in eval_files.items():
        recs = load_jsonl(p)
        eval_records[name] = recs
        eval_shingles[name] = [shingle_4gram(extract_story_question(r)) for r in recs]
        print(f"  {name}: {len(recs)}")

    # 2) Load all train sources
    train_files = [
        ROOT / "data/tom/tom_train_stage14_weighted.jsonl",
        ROOT / "data/tom/raw/hitom_train_1200.jsonl",
        ROOT / "data/tom/raw/synth_eu_emotion_v32.jsonl",
        ROOT / "data/tom/raw/synth_eu_cause_v32.jsonl",
        ROOT / "data/tom/raw/synth_ea_v32.jsonl",
        ROOT / "data/tom/raw/synth_socialiqa_v32.jsonl",
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

    # 3) Dedup intra-train (exact) + dedup vs eval (MinHash 4-gram, threshold 0.6)
    seen_hash: set[str] = set()
    leakage = []
    keep: list[dict] = []
    for r in train_records:
        text = extract_story_question(r)
        h = text_hash(text)
        if h in seen_hash:
            continue
        # leakage check vs eval
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

    # Shuffle for training
    random.seed(42)
    random.shuffle(keep)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote train → {out_train}")

    # 4) Carve validation slices from eval sets (DISJOINT from training already
    #    by construction — eval sets are not in train sources). We just convert
    #    them to the same messages format for ROLL validation.
    def to_train_msg_format(rec: dict, n_options_override: int | None = None) -> dict:
        options = rec["options"]
        n = n_options_override or len(options)
        letters = ", ".join(chr(ord("A") + i) for i in range(n))
        sys_p = (
            "You are a careful reader answering a multiple-choice question. "
            "Read the story and the question carefully, then output ONLY your final answer "
            f"in the format \\boxed{{X}} where X is one of {letters}. "
            "Do not include any explanation, reasoning, or extra text."
        )
        opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options[:n]))
        story = rec.get("story", "")
        user_p = (f"Story:\n{story}\n\n" if story else "") + f"Question: {rec['question']}\n{opts_block}"
        return {
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
            "ground_truth": rec["gold"],
            "tag": "tom_mcq",
            "source": rec.get("source", "?"),
            "language": rec.get("language", "en"),
            "task": rec.get("task", "?"),
            "question_id": rec["question_id"],
        }

    # Hi-ToM val 200: 40 per order × 5 orders
    hitom_eval = eval_records["hitom"]
    by_order: dict[str, list[dict]] = {}
    for r in hitom_eval:
        by_order.setdefault(r.get("task", "?"), []).append(r)
    rng = random.Random(42)
    hitom_val = []
    for k in sorted(by_order):
        rng.shuffle(by_order[k])
        hitom_val.extend(by_order[k][:40])
    with open(out_val_hitom, "w") as f:
        for r in hitom_val:
            f.write(json.dumps(to_train_msg_format(r), ensure_ascii=False) + "\n")
    print(f"Wrote {len(hitom_val)} hitom val → {out_val_hitom}")

    # EmoBench EU_emotion val 100
    eu_emotion = [r for r in eval_records["emobench"] if r.get("task") == "EU_emotion"]
    rng2 = random.Random(43)
    rng2.shuffle(eu_emotion)
    eu_val = eu_emotion[:100]
    with open(out_val_emobench, "w") as f:
        for r in eu_val:
            f.write(json.dumps(to_train_msg_format(r), ensure_ascii=False) + "\n")
    print(f"Wrote {len(eu_val)} EU_emotion val → {out_val_emobench}")

    # 5) Report
    from collections import Counter
    src_dist = Counter(r.get("source", "?") for r in keep)
    task_dist = Counter(r.get("task", "?") for r in keep)
    report = {
        "total_train": len(keep),
        "leakage_dropped": len(leakage),
        "leakage_examples": leakage[:20],
        "source_dist": dict(src_dist),
        "task_dist": dict(task_dist),
        "val_hitom": len(hitom_val),
        "val_eu_emotion": len(eu_val),
    }
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport → {out_report}")
    print("Source distribution:")
    for s, c in src_dist.most_common():
        print(f"  {s:<35} {c}")
    print("Task distribution (top 15):")
    for t, c in task_dist.most_common(15):
        print(f"  {t:<35} {c}")


if __name__ == "__main__":
    main()
