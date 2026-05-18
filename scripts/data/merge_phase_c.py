"""Build Stage8 train data: replace phase_a with phase_c (style-matched).

Stage7 lesson: GPT-5.5 phase_a synth was high quality but stylistically
mismatched (8-12 sentence literary stories vs ToMBench's 5-7 sentence
direct narrative). Phase C re-synthesized 1200 records with explicit
5-7 sentence style constraint.

Stage8 build:
  base:        7259 (post-cleanup tom_train, BEFORE phase_a/phase_b)
  + phase_c:   1200 (style-matched, 7 categories, 50/50 EN/ZH)
  + phase_b_zh: 800 (Chinese ToM, kept since ZH coverage was weak)
  TOTAL:      ~9259 (vs stage7's 9559)

This drops stage7's 1500 phase_a (stylistically off) and replaces with
1200 phase_c (style-matched, more focused on HOT failure categories).

Output: data/tom/tom_train.jsonl  (overwrites stage7's, with backup)
"""
import json
import shutil
from pathlib import Path
from datasketch import MinHash, MinHashLSH

NUM_PERM = 128


def _4grams(text):
    toks = text.split()
    if len(toks) < 4: return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+4]) for i in range(len(toks) - 3)}


def jaccard(a, b):
    A, B = _4grams(a), _4grams(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


def mh(text):
    m = MinHash(num_perm=NUM_PERM)
    for g in _4grams(text): m.update(g.encode("utf-8"))
    return m


def text_train(r):
    return next((m["content"] for m in r["messages"] if m["role"] == "user"), "")


def text_raw(r):
    return f"{r.get('story','')} {r.get('question','')} {r.get('opt_a','')} {r.get('opt_b','')} {r.get('opt_c','')} {r.get('opt_d','')}"


def main():
    train_path = Path("data/tom/tom_train.jsonl")
    pre_phase_a_backup = Path("data/tom/tom_train_PRE_PHASE_A_BACKUP.jsonl")
    eval_path = Path("data/tom/tombench_eval.jsonl")
    phase_c = Path("data/tom/raw/synth_gpt55_phase_c.jsonl")
    phase_b = Path("data/tom/raw/synth_gpt55_phase_b_zh.jsonl")
    backup = Path("data/tom/tom_train_PRE_PHASE_C_BACKUP.jsonl")

    # backup current state (post-phase_a/b stage7 train data)
    shutil.copyfile(train_path, backup)
    print(f"backed up current train -> {backup}")

    # restore pre-phase-a state as base
    shutil.copyfile(pre_phase_a_backup, train_path)
    base = []
    with train_path.open() as f:
        for line in f:
            base.append(json.loads(line))
    print(f"restored base from pre-phase-a backup: {len(base)} records")

    # Index eval for leakage check
    print("indexing eval ...")
    eval_recs = [json.loads(l) for l in eval_path.open()]
    lsh_eval = MinHashLSH(threshold=0.6, num_perm=NUM_PERM)
    eval_text = {}
    for r in eval_recs:
        qid = r["question_id"]
        txt = text_train(r)
        lsh_eval.insert(qid, mh(txt))
        eval_text[qid] = txt

    # Index base for internal dedupe
    print("indexing base ...")
    lsh_train = MinHashLSH(threshold=0.7, num_perm=NUM_PERM)
    train_text = {}
    for r in base:
        qid = r["question_id"]
        txt = text_train(r)
        lsh_train.insert(qid, mh(txt))
        train_text[qid] = txt

    # Build user prompts (need formatters)
    from scripts.eval.run_tombench import SYSTEM_PROMPT_DIRECT, build_user_prompt_en, build_user_prompt_zh

    new_records = [json.loads(l) for l in phase_c.open()]
    new_records += [json.loads(l) for l in phase_b.open()]
    print(f"loaded {len(new_records)} new records (phase_c 1200 + phase_b_zh 800)")

    survivors = []
    leak_drops = []
    dup_drops = []
    max_eval_j = 0.0
    for r in new_records:
        txt = text_raw(r)
        m_ = mh(txt)
        # Leakage
        cands = lsh_eval.query(m_)
        max_j = 0.0; worst = None
        for qid in cands:
            j = jaccard(txt, eval_text[qid])
            if j > max_j: max_j = j; worst = qid
        if max_j > max_eval_j: max_eval_j = max_j
        if max_j > 0.6:
            leak_drops.append({"qid": r["question_id"], "max_j": max_j, "eval": worst})
            continue
        # Dedup
        cands_t = lsh_train.query(m_)
        is_dup = False
        for qid in cands_t:
            if jaccard(txt, train_text[qid]) > 0.7:
                dup_drops.append({"qid": r["question_id"], "dup_with": qid})
                is_dup = True
                break
        if is_dup: continue
        # Format
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
        train_text[r["question_id"]] = user_text
        lsh_train.insert(r["question_id"], mh(user_text))

    print(f"new survivors: {len(survivors)} (leak {len(leak_drops)}, dup {len(dup_drops)})")
    print(f"max Jaccard vs eval: {max_eval_j:.3f}")

    total = base + survivors
    with train_path.open("w") as fp:
        for r in total:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(total)} -> {train_path}")

    Path("data/tom/merge_phase_c_report.json").write_text(json.dumps({
        "base": len(base), "added": len(survivors), "after": len(total),
        "leak_drops": len(leak_drops), "dup_drops": len(dup_drops),
        "max_eval_jaccard": round(max_eval_j, 4),
    }, indent=2))
    print("report -> data/tom/merge_phase_c_report.json")


if __name__ == "__main__":
    main()
