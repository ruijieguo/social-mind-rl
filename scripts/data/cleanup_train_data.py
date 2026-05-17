"""Clean training data based on GPT-5.5 audit findings.

Audit results (per-source, n=30 each):
  exploretom         3% high  /  89% low+harmful  -> DROP all
  exploretom_zh      3% high  /  86% low+harmful  -> DROP all
  simpletom         60% high  /   6% low+harmful  -> keep
  simpletom_zh      36% high  /  39% low+harmful  -> DROP all
  synth (flash)     96% high  /   0% low+harmful  -> keep
  synth_zh          90% high  /   0% low+harmful  -> keep
  synth_phase1     50% high  /  43% low+harmful  -> filter (drop low/harmful, keep high/medium)

For synth_phase1 we filter at record level using the per-record audit
results in output/analysis/gpt55_train_audit.jsonl. We only have audit
data for 30 of ~990 phase1 records, so we conservatively keep all
synth_phase1 records that weren't AUDITED-AS-low-or-harmful, and drop
the rest pending a full audit.

NEW source we add: synth_gpt55 (the high-quality GPT-5.5 1400 records).

Output: updated tom_train.jsonl + tom_train_4k.jsonl
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from collections import Counter


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    # Load tom_train (the merged result)
    train = load_jsonl("data/tom/tom_train.jsonl")
    print(f"Original tom_train.jsonl: {len(train)} records")
    print("Original by source:")
    src_counts = Counter(r.get("source") for r in train)
    for k, v in src_counts.most_common(): print(f"  {k}: {v}")
    print()

    # Drop policy
    DROP_SOURCES = {"exploretom", "exploretom_zh", "simpletom_zh"}

    # synth_phase1 filter: keep only those NOT marked as low/harmful in audit
    audit_low_harmful_qids = set()
    audit_path = Path("output/analysis/gpt55_train_audit.jsonl")
    if audit_path.exists():
        for line in audit_path.open():
            a = json.loads(line)
            if a.get("source", "") == "synth_phase1":
                if a["audit"].get("training_value") in ("low", "harmful"):
                    audit_low_harmful_qids.add(a["qid"])
        print(f"phase1 records explicitly marked low/harmful by GPT-5.5: {len(audit_low_harmful_qids)}")
    # Note: only ~30/990 phase1 audited; broader cleanup would require full audit.
    # For now we drop the audited bad ones and keep the rest.

    cleaned = []
    dropped_by_source = Counter()
    for r in train:
        src = r.get("source")
        if src in DROP_SOURCES:
            dropped_by_source[src] += 1
            continue
        if src == "synth_phase1" and r.get("question_id") in audit_low_harmful_qids:
            dropped_by_source["synth_phase1_audited_bad"] += 1
            continue
        cleaned.append(r)

    print(f"\nDropped by policy:")
    for k, v in dropped_by_source.most_common(): print(f"  {k}: {v}")
    print(f"\nAfter cleanup (before adding gpt55): {len(cleaned)} records")

    # Add GPT-5.5 synth records — convert from raw schema to messages format
    gpt55_path = Path("data/tom/raw/synth_gpt55.jsonl")
    if gpt55_path.exists():
        from scripts.eval.run_tombench import SYSTEM_PROMPT_DIRECT, build_user_prompt_en, build_user_prompt_zh
        added = 0
        for line in gpt55_path.open():
            r = json.loads(line)
            # Convert to chat format
            user_prompt_fn = build_user_prompt_en if r["language"] == "en" else build_user_prompt_zh
            user_content = user_prompt_fn(
                story=r["story"], question=r["question"],
                opt_a=r["opt_a"], opt_b=r["opt_b"], opt_c=r["opt_c"], opt_d=r["opt_d"],
            )
            new_rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
                    {"role": "user", "content": user_content},
                ],
                "ground_truth": r["gold"],
                "tag": "tom_mcq",
                "source": "synth_gpt55",
                "language": r["language"],
                "task": r["task"],
                "question_id": r["question_id"],
            }
            cleaned.append(new_rec)
            added += 1
        print(f"\nAdded {added} GPT-5.5 synth records")

    print(f"\nFinal cleaned tom_train.jsonl: {len(cleaned)} records")

    out_train = Path("data/tom/tom_train_cleaned.jsonl")
    with out_train.open("w") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_train}")

    # 4k subset for stage1-style verification
    rng = random.Random(42)
    sample = list(cleaned)
    rng.shuffle(sample)
    out_4k = Path("data/tom/tom_train_4k_cleaned.jsonl")
    with out_4k.open("w") as f:
        for r in sample[:4000]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_4k}")

    print("\nFinal distribution:")
    final_src = Counter(r.get("source") for r in cleaned)
    for k, v in final_src.most_common(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
