"""Three-way error analysis: 14B-tom-stage6 vs deepseek-v4-pro vs GPT-5.5 on full 5718.

Goal: identify which 14B errors are REAL improvement opportunities (where
GPT-5.5 is right) vs hard-ceiling errors (where even GPT-5.5 is wrong).

This separates "training problems we can fix" from "label problems / genuinely
hard questions" — the difference matters because:
  - For real improvements: more/better training data targeted at the task
  - For hard ceiling: the eval-set itself caps us, no amount of training helps

Outputs:
  output/analysis/threeway_errors.md   - summary tables
  output/analysis/threeway_catchable.json  - actionable error list per task
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def by_qid(records, model, proto="direct"):
    return {r["question_id"]: r for r in records if r.get("model") == model and r.get("protocol") == proto}


def main():
    s14 = load("output/eval/14b_full5718.json")
    s14_s6 = load("output/eval/stage6_full5718.json")
    s8_s6 = load("output/eval/8b_stage6_full5718.json")
    s8_s1 = load("output/eval/final_full5718.json")
    ds = load("output/eval/deepseek_full5718.json")
    gpt5 = load("output/eval/gpt-5.5_full5718.json")

    s14_s6_idx = by_qid(s14_s6, "qwen3-14b-tom-stage6")
    s8_s1_idx = by_qid(s8_s1, "qwen3-8b-tom")
    s8_s6_idx = by_qid(s8_s6, "qwen3-8b-tom-stage6")
    ds_idx = by_qid(ds, "deepseek-v4-pro")
    gpt_idx = by_qid(gpt5, "gpt-5.5")
    s14_idx = by_qid(s14, "qwen3-14b-tom")

    print("# Three-way error decomposition for 14B stage6\n")

    # ===== 1. Where 14B-stage6 errs, who else gets it right? =====
    print("## 1. 14B-stage6 errors by who-also-got-it-wrong\n")
    print("Out of all 14B-stage6 errors (n=1354), classify:")
    print("- ALL_THREE_WRONG: GPT-5.5 + deepseek also wrong → hard ceiling / label noise")
    print("- ONLY_GPT5_RIGHT: GPT-5.5 right, deepseek wrong → 14B can definitely catch")
    print("- ONLY_DS_RIGHT:  deepseek right, GPT-5.5 wrong → 14B can definitely catch")
    print("- BOTH_RIGHT_BUT_14B_WRONG: both right → catchable, the high-confidence target")
    print()

    error_types = defaultdict(lambda: defaultdict(int))
    catchable_targets = defaultdict(list)  # by task

    n_total = 0
    for qid, r14 in s14_s6_idx.items():
        if r14["correct"]:
            continue
        n_total += 1
        ds_r = ds_idx.get(qid, {})
        gpt_r = gpt_idx.get(qid, {})
        ds_correct = ds_r.get("correct", False)
        gpt_correct = gpt_r.get("correct", False)
        task = r14["task"]
        if not ds_correct and not gpt_correct:
            cat = "all_three_wrong"
        elif gpt_correct and not ds_correct:
            cat = "only_gpt5_right"
        elif ds_correct and not gpt_correct:
            cat = "only_ds_right"
        elif gpt_correct and ds_correct:
            cat = "both_right_14b_wrong"
        error_types[task][cat] += 1
        if cat == "both_right_14b_wrong":
            # high-confidence catchable
            catchable_targets[task].append({
                "qid": qid, "lang": r14["language"], "gold": r14["gold"],
                "our_pred": r14["pred"], "ds_pred": ds_r.get("pred"), "gpt_pred": gpt_r.get("pred"),
            })

    print("| Task | Total errors | hard ceiling | only_gpt | only_ds | both_right_14b_wrong (HOT) |")
    for task in sorted(error_types):
        d = error_types[task]
        total = sum(d.values())
        print(f"| {task} | {total} | {d.get('all_three_wrong', 0)} ({d.get('all_three_wrong', 0)*100//total}%) | {d.get('only_gpt5_right', 0)} | {d.get('only_ds_right', 0)} | {d.get('both_right_14b_wrong', 0)} ({d.get('both_right_14b_wrong', 0)*100//total}%) |")
    print()

    # Total summary
    n_hard = sum(d.get('all_three_wrong', 0) for d in error_types.values())
    n_hot = sum(d.get('both_right_14b_wrong', 0) for d in error_types.values())
    print(f"Total 14B-stage6 errors: {n_total}")
    print(f"  hard ceiling (all 3 wrong):           {n_hard} ({n_hard*100//n_total}%)")
    print(f"  catchable HOT (both other 2 right):   {n_hot} ({n_hot*100//n_total}%)")
    print(f"  closing all HOT would gain:           {n_hot/5718*100:.2f}pp")
    print()

    # ===== 2. EN vs ZH per task gap =====
    print("## 2. EN vs ZH per-task gaps (14B stage6 → GPT-5.5)\n")
    def acc(idx, task, lang):
        recs = [r for r in idx.values() if r["task"] == task and r["language"] == lang]
        return sum(r["correct"] for r in recs) / len(recs) if recs else 0
    tasks = sorted(set(r["task"] for r in s14_s6_idx.values()))
    print("| Task | s6 EN | s6 ZH | ZH-EN | gpt EN | gpt ZH | s6 vs gpt EN | s6 vs gpt ZH |")
    for t in tasks:
        s6_en = acc(s14_s6_idx, t, "en"); s6_zh = acc(s14_s6_idx, t, "zh")
        gpt_en = acc(gpt_idx, t, "en"); gpt_zh = acc(gpt_idx, t, "zh")
        print(f"| {t} | {s6_en:.3f} | {s6_zh:.3f} | {s6_zh-s6_en:+.3f} | {gpt_en:.3f} | {gpt_zh:.3f} | {s6_en-gpt_en:+.3f} | {s6_zh-gpt_zh:+.3f} |")
    print()

    # ===== 3. Where GPT-5.5 specifically beats deepseek =====
    print("## 3. Where GPT-5.5 specifically beats deepseek (signal of headroom)\n")
    print("Tasks where GPT-5.5 outperforms deepseek = where there's measurable signal beyond what deepseek gives us.\n")
    print("| Task | deepseek | gpt5 | gpt-ds gap |")
    for t in tasks:
        ds_acc = sum(r["correct"] for r in ds_idx.values() if r["task"] == t) / sum(1 for r in ds_idx.values() if r["task"] == t)
        gpt_acc = sum(r["correct"] for r in gpt_idx.values() if r["task"] == t) / sum(1 for r in gpt_idx.values() if r["task"] == t)
        print(f"| {t} | {ds_acc:.3f} | {gpt_acc:.3f} | {gpt_acc-ds_acc:+.3f} |")
    print()

    # ===== 4. 14B stage6 vs stage1 progression analysis =====
    print("## 4. Did stage6 actually improve over stage1, or just shuffle errors?\n")
    s14_s1_correct_s6_wrong = 0
    s14_s1_wrong_s6_correct = 0
    for qid, r in s14_idx.items():
        r6 = s14_s6_idx.get(qid)
        if not r6: continue
        if r["correct"] and not r6["correct"]:
            s14_s1_correct_s6_wrong += 1
        elif not r["correct"] and r6["correct"]:
            s14_s1_wrong_s6_correct += 1
    print(f"  s1 right → s6 wrong: {s14_s1_correct_s6_wrong}")
    print(f"  s1 wrong → s6 right: {s14_s1_wrong_s6_correct}")
    print(f"  net delta: {s14_s1_wrong_s6_correct - s14_s1_correct_s6_wrong} (= +{(s14_s1_wrong_s6_correct - s14_s1_correct_s6_wrong)*100/5718:.2f}pp)")
    print()

    # save catchable HOT list
    out_path = Path("output/analysis/threeway_catchable_hot.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catchable_targets, ensure_ascii=False, indent=2))
    total_hot = sum(len(v) for v in catchable_targets.values())
    print(f"Saved {total_hot} HOT catchable error qids to {out_path}")


if __name__ == "__main__":
    main()
