"""Deep bad-case analysis across baseline / stage1 / stage2 to inform the
next iteration of data + training plans.

Cross-tabs we care about:
- Per-task accuracy: baseline qwen-nt | stage1 | stage2 | deepseek (direct)
- Question-level: which qids did stage1 get right that stage2 lost (regressions)
- Question-level: which qids does deepseek get right that NO open model gets
- Predicted-letter histogram on errors (mode collapse?)
- Story length / response length stats on errors vs corrects
- The 'gap closeable' set: where deepseek is right and qwen still wrong
"""
from __future__ import annotations
import json
from collections import defaultdict, Counter
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def index_by_qid(records, model_filter=None, proto="direct"):
    out = {}
    for r in records:
        if model_filter and r["model"] != model_filter:
            continue
        if r["protocol"] != proto:
            continue
        out[r["question_id"]] = r
    return out


def pct(num, denom):
    return f"{100*num/denom:.1f}%" if denom else "-"


def main():
    print("# Stage1/2 vs deepseek bad-case analysis")
    print()

    base = load("output/eval/baseline_combined.json")
    s1 = load("output/eval/final_full5718.json")
    s2 = load("output/eval/stage2_full5718.json")

    nt = index_by_qid(base, "qwen3-8b-nt", "direct")            # full 5718
    ds = index_by_qid(base, "deepseek-v4-pro", "direct")        # 500 subset
    s1_idx = index_by_qid(s1, "qwen3-8b-tom", "direct")         # full 5718
    s2_idx = index_by_qid(s2, "qwen3-8b-tom-stage2", "direct")  # full 5718

    print(f"baseline qwen-nt: {len(nt)} | stage1: {len(s1_idx)} | stage2: {len(s2_idx)} | deepseek: {len(ds)}")
    print()

    # ===== 1. Per-task accuracy 4-way =====
    print("## 1. Per-task accuracy (direct, full 5718 unless deepseek)")
    print()
    print("| Task | baseline | stage1 | stage2 | deepseek (n=500) |")
    print("|---|---|---|---|---|")
    tasks = sorted({r["task"] for r in nt.values()})
    for t in tasks:
        def acc(idx, t):
            recs = [r for r in idx.values() if r["task"] == t]
            return sum(r["correct"] for r in recs) / len(recs) if recs else 0
        print(f"| {t} | {acc(nt, t):.4f} | {acc(s1_idx, t):.4f} | {acc(s2_idx, t):.4f} | {acc(ds, t):.4f} |")
    print()

    # ===== 2. Stage1 vs Stage2 regressions =====
    print("## 2. Stage1→Stage2 regressions (s1 correct, s2 wrong)")
    print()
    regressions = []
    improvements = []
    for qid, r1 in s1_idx.items():
        r2 = s2_idx.get(qid)
        if not r2:
            continue
        if r1["correct"] and not r2["correct"]:
            regressions.append(qid)
        elif not r1["correct"] and r2["correct"]:
            improvements.append(qid)
    print(f"  - regressions (s1 ✓ → s2 ✗): {len(regressions)}")
    print(f"  - improvements (s1 ✗ → s2 ✓): {len(improvements)}")
    print(f"  - net Δ: {len(improvements) - len(regressions)}")
    print()

    # break down regressions by task
    reg_by_task = Counter(s1_idx[q]["task"] for q in regressions)
    imp_by_task = Counter(s1_idx[q]["task"] for q in improvements)
    print("  per-task breakdown:")
    print()
    print("  | Task | s1→s2 lost | s1→s2 gained | net |")
    print("  |---|---|---|---|")
    for t in tasks:
        L = reg_by_task.get(t, 0)
        G = imp_by_task.get(t, 0)
        print(f"  | {t} | {L} | {G} | {G-L:+d} |")
    print()

    # ===== 3. Closeable gap on deepseek subset =====
    print("## 3. Closeable gap on 500-subset (where deepseek ✓ but stage2 ✗)")
    print()
    closeable_s2 = []   # deepseek right, stage2 wrong
    both_wrong = []     # neither got it
    only_s2 = []        # stage2 right, deepseek wrong (we already win)
    both_right = []
    for qid, d in ds.items():
        r2 = s2_idx.get(qid)
        if not r2:
            continue
        if d["correct"] and not r2["correct"]:
            closeable_s2.append(qid)
        elif not d["correct"] and not r2["correct"]:
            both_wrong.append(qid)
        elif not d["correct"] and r2["correct"]:
            only_s2.append(qid)
        else:
            both_right.append(qid)
    n = len(ds)
    print(f"  - both correct:                       {len(both_right)} ({pct(len(both_right), n)})")
    print(f"  - only deepseek correct (CLOSEABLE):  {len(closeable_s2)} ({pct(len(closeable_s2), n)})")
    print(f"  - only stage2 correct (bonus):        {len(only_s2)} ({pct(len(only_s2), n)})")
    print(f"  - both wrong (hard ceiling):          {len(both_wrong)} ({pct(len(both_wrong), n)})")
    print(f"  - if we close all {len(closeable_s2)}, gain = {len(closeable_s2)/n*100:.1f}pp")
    print()

    # Closeable breakdown by task
    close_by_task = Counter(ds[q]["task"] for q in closeable_s2)
    both_wrong_by_task = Counter(ds[q]["task"] for q in both_wrong)
    n_by_task = Counter(d["task"] for d in ds.values())
    print("  CLOSEABLE by task (deepseek ✓ stage2 ✗ on 500 subset):")
    print()
    print("  | Task | n_500 | closeable | both wrong (ceiling) | closeable_rate |")
    print("  |---|---|---|---|---|")
    for t in tasks:
        cc = close_by_task.get(t, 0)
        bw = both_wrong_by_task.get(t, 0)
        nt = n_by_task.get(t, 0)
        rate = cc / nt if nt else 0
        print(f"  | {t} | {nt} | {cc} | {bw} | {rate*100:.1f}% |")
    print()

    # ===== 4. Per-task gap to deepseek (subset 500 numbers) =====
    print("## 4. Per-task gap and theoretical max upside on 500 subset")
    print()
    print("| Task | n | s2 acc | ds acc | gap | full-close-ceiling |")
    print("|---|---|---|---|---|---|")
    total_weighted_close = 0
    total_n = 0
    for t in tasks:
        ds_r = [r for r in ds.values() if r["task"] == t]
        s2_r = [s2_idx[q] for q in ds if s2_idx.get(q) and ds[q]["task"] == t]
        if not ds_r or not s2_r:
            continue
        ds_acc = sum(r["correct"] for r in ds_r) / len(ds_r)
        s2_acc = sum(r["correct"] for r in s2_r) / len(s2_r)
        gap = ds_acc - s2_acc
        full_close_pp = gap * len(ds_r) / n
        if gap > 0:
            total_weighted_close += gap * len(ds_r)
        total_n += len(ds_r)
        print(f"| {t} | {len(ds_r)} | {s2_acc:.4f} | {ds_acc:.4f} | {gap:+.4f} | {full_close_pp*100:+.2f}pp |")
    print(f"\nweighted total upside if all positive gaps closed: {total_weighted_close/total_n*100:.2f}pp")
    print()

    # ===== 5. Predicted-letter distribution on errors =====
    print("## 5. Predicted-letter distribution on stage2 errors (mode collapse?)")
    print()
    s2_errors = [r for r in s2_idx.values() if not r["correct"]]
    s2_correct = [r for r in s2_idx.values() if r["correct"]]
    err_pred = Counter(r["pred"] for r in s2_errors)
    err_gold = Counter(r["gold"] for r in s2_errors)
    cor_gold = Counter(r["gold"] for r in s2_correct)
    print(f"  errors: {len(s2_errors)}")
    print(f"  pred dist on errors:  {dict(err_pred)}")
    print(f"  gold dist on errors:  {dict(err_gold)}")
    print(f"  gold dist on corrects:{dict(cor_gold)}")
    print()
    # confusion matrix
    print("  Confusion matrix on errors (gold → pred):")
    print("  | gold↓ pred→ | A | B | C | D | None |")
    print("  |---|---|---|---|---|---|")
    for g in "ABCD":
        row = [g]
        for p in "ABCD":
            row.append(str(sum(1 for r in s2_errors if r["gold"] == g and r["pred"] == p)))
        row.append(str(sum(1 for r in s2_errors if r["gold"] == g and (r["pred"] or "") not in "ABCD")))
        print("  | " + " | ".join(row) + " |")
    print()

    # ===== 6. Response style on errors vs corrects =====
    print("## 6. Response patterns on stage2 errors")
    print()
    def has_think(r):
        raw = (r.get("raw_responses") or [""])[0]
        return "<think>" in raw or "\\boxed" in raw

    has_think_err = sum(1 for r in s2_errors if has_think(r))
    has_think_cor = sum(1 for r in s2_correct if has_think(r))
    print(f"  errors w/ <think>: {has_think_err}/{len(s2_errors)} ({pct(has_think_err, len(s2_errors))})")
    print(f"  corrects w/ <think>: {has_think_cor}/{len(s2_correct)} ({pct(has_think_cor, len(s2_correct))})")
    print()

    def resp_len(r):
        return len((r.get("raw_responses") or [""])[0])
    err_lens = [resp_len(r) for r in s2_errors]
    cor_lens = [resp_len(r) for r in s2_correct]
    print(f"  err response length: mean={sum(err_lens)/len(err_lens):.0f} max={max(err_lens)}")
    print(f"  cor response length: mean={sum(cor_lens)/len(cor_lens):.0f} max={max(cor_lens)}")
    print()

    # ===== 7. Both-wrong analysis (hard ceiling) =====
    print("## 7. Hard ceiling — both deepseek and stage2 wrong")
    print()
    print("  These represent the limit of MCQ-as-eval / ToMBench labeling difficulty.")
    print(f"  Total both-wrong: {len(both_wrong)} ({pct(len(both_wrong), n)})")
    print()
    print("  by task:")
    print()
    print("  | Task | n_500 | both-wrong | rate |")
    print("  |---|---|---|---|")
    for t in tasks:
        bw = both_wrong_by_task.get(t, 0)
        nt = n_by_task.get(t, 0)
        print(f"  | {t} | {nt} | {bw} | {pct(bw, nt)} |")
    print()


if __name__ == "__main__":
    main()
