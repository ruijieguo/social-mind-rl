"""Analyze baseline errors to identify training priorities."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    records = json.loads(Path("output/eval/baseline_combined.json").read_text())
    print(f"Total records: {len(records)}\n")

    # Filter to qwen3-8b-nt direct (our training start state)
    nt_direct = [r for r in records if r["model"] == "qwen3-8b-nt" and r["protocol"] == "direct"]
    nt_errors = [r for r in nt_direct if not r["correct"]]
    deepseek_direct = [r for r in records if r["model"] == "deepseek-v4-pro" and r["protocol"] == "direct"]

    qid_to_deepseek_correct = {r["question_id"]: r["correct"] for r in deepseek_direct}

    # Categorize errors
    # A: both qwen and deepseek wrong (hard for everyone) - low training priority
    # B: deepseek right, qwen wrong (gap we can close) - HIGH priority
    # C: deepseek wrong but qwen right (we already beat it) - bonus
    both_wrong = 0
    we_can_close = 0  # deepseek correct, qwen wrong
    only_deepseek_wrong = 0  # qwen correct, deepseek wrong
    both_correct = 0

    for r in nt_direct:
        qid = r["question_id"]
        qwen_correct = r["correct"]
        deepseek_correct = qid_to_deepseek_correct.get(qid, None)
        if deepseek_correct is None:
            continue
        if qwen_correct and deepseek_correct:
            both_correct += 1
        elif qwen_correct and not deepseek_correct:
            only_deepseek_wrong += 1
        elif not qwen_correct and deepseek_correct:
            we_can_close += 1
        else:
            both_wrong += 1
    n = both_correct + only_deepseek_wrong + we_can_close + both_wrong

    print(f"Error categorization (qwen3-8b-nt direct vs deepseek-v4-pro direct), n={n}")
    print(f"  Both correct:                {both_correct:4d} ({100*both_correct/n:5.1f}%)")
    print(f"  Only deepseek correct (gap): {we_can_close:4d} ({100*we_can_close/n:5.1f}%)  ← training upside")
    print(f"  Only qwen correct (bonus):   {only_deepseek_wrong:4d} ({100*only_deepseek_wrong/n:5.1f}%)")
    print(f"  Both wrong (hard):           {both_wrong:4d} ({100*both_wrong/n:5.1f}%)")
    print()

    # Per-task gap analysis
    print(f"Per-task gap analysis (% of errors where deepseek is right but qwen wrong):")
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "we_can_close": 0, "both_wrong": 0, "qwen_correct": 0})
    for r in nt_direct:
        qid = r["question_id"]
        qwen_correct = r["correct"]
        deepseek_correct = qid_to_deepseek_correct.get(qid, None)
        if deepseek_correct is None:
            continue
        task = r.get("task", "?")
        by_task[task]["n"] += 1
        if qwen_correct:
            by_task[task]["qwen_correct"] += 1
        if not qwen_correct and deepseek_correct:
            by_task[task]["we_can_close"] += 1
        if not qwen_correct and not deepseek_correct:
            by_task[task]["both_wrong"] += 1
    print(f"  {'task':25s}  {'n':>4s}  {'qwen-acc':>10s}  {'gap-to-close':>14s}  {'both-wrong':>11s}")
    for task in sorted(by_task, key=lambda t: -by_task[t]["we_can_close"]):
        s = by_task[task]
        qwen_acc = s["qwen_correct"] / s["n"]
        gap = s["we_can_close"] / s["n"]
        bw = s["both_wrong"] / s["n"]
        print(f"  {task:25s}  {s['n']:4d}  {qwen_acc:9.1%}  {gap:13.1%}  {bw:10.1%}")
    print()

    # Per-language gap
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "qwen_correct": 0, "deepseek_correct": 0})
    for r in nt_direct:
        qid = r["question_id"]
        lang = r.get("language", "?")
        deepseek_correct = qid_to_deepseek_correct.get(qid, None)
        if deepseek_correct is None:
            continue
        by_lang[lang]["n"] += 1
        if r["correct"]:
            by_lang[lang]["qwen_correct"] += 1
        if deepseek_correct:
            by_lang[lang]["deepseek_correct"] += 1
    print(f"Per-language overall:")
    for lang, s in sorted(by_lang.items()):
        qwen_acc = s["qwen_correct"] / s["n"]
        ds_acc = s["deepseek_correct"] / s["n"]
        gap_pp = (ds_acc - qwen_acc) * 100
        print(f"  {lang}: n={s['n']}  qwen={qwen_acc:.3f}  deepseek={ds_acc:.3f}  gap={gap_pp:+.1f}pp")

    # Predicted-letter distribution (reward hacking detection)
    print(f"\nqwen3-8b-nt direct: predicted-letter distribution (should be uniform if not hacking):")
    preds = Counter(r["pred"] for r in nt_direct if r["pred"] in {"A", "B", "C", "D"})
    none_preds = sum(1 for r in nt_direct if r["pred"] is None or r["pred"] not in {"A", "B", "C", "D"})
    total = sum(preds.values()) + none_preds
    for L in "ABCD":
        v = preds.get(L, 0)
        print(f"  {L}: {v:4d} ({100*v/total:5.1f}%)")
    print(f"  None/invalid: {none_preds} ({100*none_preds/total:5.1f}%)")


if __name__ == "__main__":
    main()
