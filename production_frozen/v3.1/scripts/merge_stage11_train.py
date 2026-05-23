"""
Stage 11 Track E: Merge stage 8 base data + Track B (ExploreToM) + Track C (HOT synth)
into a single training file.

Usage:
    python scripts/data/merge_stage11_train.py \
        --base data/tom/tom_train.jsonl \
        --add data/tom/raw/synth_gpt55_phase_d_hot.jsonl \
              data/tom/raw/exploretom_v2.jsonl \
        --output data/tom/tom_train_stage11.jsonl \
        --leakage-check
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def to_messages(rec, language=None):
    """Convert ToMBench-format record to messages format used by stage 8 training."""
    if "messages" in rec:
        return rec
    lang = language or rec.get("language", "en")
    if lang == "zh":
        sys_prompt = "你是一个细心的读者，请回答下列心智推理多项选择题。请只回答字母（A、B、C 或 D）。"
        prefix = "故事：\n"
        q_prefix = "问题："
    else:
        sys_prompt = ("You are a careful reader answering a multiple-choice theory-of-mind "
                      "question. Reply with ONLY the letter (A, B, C, or D).")
        prefix = "Story:\n"
        q_prefix = "Question:"
    user = (
        f"{prefix}{rec['story']}\n\n"
        f"{q_prefix} {rec['question']}\n"
        f"A. {rec['opt_a']}\n"
        f"B. {rec['opt_b']}\n"
        f"C. {rec['opt_c']}\n"
        f"D. {rec['opt_d']}"
    )
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        "ground_truth": rec["gold"],
        "question_id": rec.get("question_id", ""),
        "task": rec.get("task", ""),
        "source": rec.get("source", ""),
        "language": lang,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/tom/tom_train.jsonl",
                    help="Stage 8 base training data (already in messages format)")
    ap.add_argument("--add", nargs="+", required=True,
                    help="New data files (ToMBench format) to add")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval-data", default="data/tom/tombench_eval.jsonl",
                    help="For leakage check")
    ap.add_argument("--no-leakage-check", action="store_true")
    args = ap.parse_args()

    base = load_jsonl(args.base) if Path(args.base).exists() else []
    base_count = len(base)
    print(f"Base ({args.base}): {base_count} records")

    seen_qids = {r.get("question_id") for r in base if r.get("question_id")}
    seen_text = set()
    for r in base:
        msgs = r.get("messages", [])
        if msgs and len(msgs) >= 2:
            # Hash the full user content for accurate dedup
            seen_text.add(hash(msgs[1]["content"]))

    new_recs = []
    for path in args.add:
        if not Path(path).exists():
            print(f"  ⚠ {path} not found, skipping")
            continue
        recs = load_jsonl(path)
        print(f"Adding {path}: {len(recs)} records")
        before = len(new_recs)
        for r in recs:
            qid = r.get("question_id")
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            converted = to_messages(r)
            text_key = hash(converted["messages"][1]["content"])
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            new_recs.append(converted)
        print(f"  added {len(new_recs) - before} (after dedup)")

    if not args.no_leakage_check and new_recs and Path(args.eval_data).exists():
        print(f"\nLeakage check vs {args.eval_data} ...")
        eval_recs = load_jsonl(args.eval_data)
        eval_text = set()
        for r in eval_recs:
            t = (r.get("story", "") + " " + r.get("question", ""))[:200]
            eval_text.add(t)
        leaked = 0
        for r in new_recs:
            content = r["messages"][1]["content"]
            for et in eval_text:
                if et and et[:80] in content:
                    leaked += 1
                    break
        print(f"  prefix-match leaked: {leaked}/{len(new_recs)}")

    all_recs = base + new_recs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total records: {len(all_recs)} (base {base_count} + new {len(new_recs)})")
    sources = Counter(r.get("source", "?") for r in all_recs)
    tasks = Counter(r.get("task", "?") for r in all_recs)
    langs = Counter(r.get("language", "?") for r in all_recs)
    print(f"Sources: {dict(sources)}")
    print(f"Tasks: {dict(tasks)}")
    print(f"Languages: {dict(langs)}")
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
