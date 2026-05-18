"""Merge Phase A synth (1500 phase_a + 800 phase_b_zh) into tom_train.jsonl.

Differs from merge_and_dedupe.py: starts from the already-cleaned tom_train.jsonl
(7259 records, post-stage6 cleanup) rather than rebuilding from raw sources.

Steps:
  1. Load existing tom_train.jsonl (7259) and tombench_eval.jsonl
  2. Load phase_a (1500) + phase_b_zh (800) = 2300 new records
  3. MinHash leakage check vs eval (drop if Jaccard 4-gram >= 0.6)
  4. Internal dedupe (drop if Jaccard >= 0.7 vs any existing or earlier-new record)
  5. Format new records with messages field
  6. Write tom_train.jsonl (target ~9559)

Output:
  data/tom/tom_train.jsonl        -- merged
  data/tom/tom_train_phase_a_backup.jsonl  -- backup of pre-phase-a state
  data/tom/merge_phase_a_report.json
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

from datasketch import MinHash, MinHashLSH


NUM_PERM = 128


def _4grams(text: str) -> set[str]:
    toks = text.split()
    if len(toks) < 4: return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+4]) for i in range(len(toks) - 3)}


def jaccard_4gram(a: str, b: str) -> float:
    A, B = _4grams(a), _4grams(b)
    if not A and not B: return 0.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


def mh(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for g in _4grams(text): m.update(g.encode("utf-8"))
    return m


def text_for_train_record(r):
    """For records in tom_train.jsonl format (with messages)."""
    user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
    return user


def text_for_raw_record(r):
    """For records in raw synth format (story/question/options at top)."""
    return f"{r.get('story','')} {r.get('question','')} {r.get('opt_a','')} {r.get('opt_b','')} {r.get('opt_c','')} {r.get('opt_d','')}"


def main():
    train_path = Path("data/tom/tom_train.jsonl")
    eval_path = Path("data/tom/tombench_eval.jsonl")
    phase_a = Path("data/tom/raw/synth_gpt55_phase_a.jsonl")
    phase_b = Path("data/tom/raw/synth_gpt55_phase_b_zh.jsonl")
    backup_path = Path("data/tom/tom_train_PRE_PHASE_A_BACKUP.jsonl")
    report_path = Path("data/tom/merge_phase_a_report.json")

    # Backup
    shutil.copyfile(train_path, backup_path)
    print(f"backed up {train_path} -> {backup_path}")

    # Load eval into MinHash LSH for leakage check
    print("indexing eval set ...")
    eval_records = []
    with eval_path.open() as f:
        for line in f:
            r = json.loads(line)
            eval_records.append(r)
    lsh_eval = MinHashLSH(threshold=0.6, num_perm=NUM_PERM)
    eval_mh_by_qid = {}
    eval_text_by_qid = {}
    for r in eval_records:
        qid = r["question_id"]
        txt = text_for_train_record(r)
        m = mh(txt)
        lsh_eval.insert(qid, m)
        eval_mh_by_qid[qid] = m
        eval_text_by_qid[qid] = txt

    # Load existing train (already-cleaned 7259) for internal dedup index
    print("indexing existing train set ...")
    existing_train = []
    with train_path.open() as f:
        for line in f:
            existing_train.append(json.loads(line))
    lsh_train = MinHashLSH(threshold=0.7, num_perm=NUM_PERM)
    train_text_by_qid = {}
    for r in existing_train:
        qid = r["question_id"]
        txt = text_for_train_record(r)
        lsh_train.insert(qid, mh(txt))
        train_text_by_qid[qid] = txt

    # Load Phase A raw records
    new_records = []
    for p in [phase_a, phase_b]:
        with p.open() as f:
            for line in f:
                new_records.append(json.loads(line))
    print(f"loaded {len(new_records)} new records (phase_a + phase_b_zh)")

    # Process: leakage check, then internal dedup
    from scripts.eval.run_tombench import SYSTEM_PROMPT_DIRECT, build_user_prompt_en, build_user_prompt_zh

    survivors = []
    dropped_leak = []
    dropped_dup = []
    max_eval_jaccard = 0.0
    for r in new_records:
        txt = text_for_raw_record(r)
        m = mh(txt)
        # Leakage check vs eval
        cands = lsh_eval.query(m)
        max_j = 0.0
        worst_qid = None
        for qid in cands:
            j = jaccard_4gram(txt, eval_text_by_qid[qid])
            if j > max_j:
                max_j = j; worst_qid = qid
        if max_j > max_eval_jaccard: max_eval_jaccard = max_j
        if max_j > 0.6:
            dropped_leak.append({"qid": r["question_id"], "max_jaccard": max_j, "eval_qid": worst_qid})
            continue
        # Internal dedup vs train (existing + new survivors)
        cands_t = lsh_train.query(m)
        is_dup = False
        for qid in cands_t:
            other = train_text_by_qid[qid]
            if jaccard_4gram(txt, other) > 0.7:
                dropped_dup.append({"qid": r["question_id"], "dup_with": qid})
                is_dup = True
                break
        if is_dup: continue
        # Survives — format as train record and add to index
        builder = build_user_prompt_zh if r["language"] == "zh" else build_user_prompt_en
        user_text = builder(
            story=r["story"], question=r["question"],
            opt_a=r["opt_a"], opt_b=r["opt_b"],
            opt_c=r["opt_c"], opt_d=r["opt_d"],
        )
        train_rec = {
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
        }
        survivors.append(train_rec)
        # Add to lsh_train for downstream dedup
        train_text_by_qid[r["question_id"]] = user_text
        lsh_train.insert(r["question_id"], mh(user_text))

    print(f"new survivors: {len(survivors)} (dropped {len(dropped_leak)} leakage, {len(dropped_dup)} dups)")
    print(f"max Jaccard vs eval: {max_eval_jaccard:.3f}")

    # Append to existing train
    total = existing_train + survivors
    with train_path.open("w") as fp:
        for r in total:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(total)} records to {train_path}")

    # Report
    report = {
        "before": len(existing_train),
        "added": len(survivors),
        "after": len(total),
        "dropped_leak": len(dropped_leak),
        "dropped_dup": len(dropped_dup),
        "max_eval_jaccard": round(max_eval_jaccard, 4),
        "dropped_leak_sample": dropped_leak[:10],
        "dropped_dup_sample": dropped_dup[:10],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
