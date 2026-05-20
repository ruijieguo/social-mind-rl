"""
Comprehensive comparison: stage 8 (3 protocols) vs deepseek-v4-pro vs GPT-5.5
on full 5718 ToMBench eval, plus per-task breakdown.

Used to guide Stage 12 data targeting and as headline result for the writeup.
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage11a", default="output/eval/stage11A_full5718.json")
    ap.add_argument("--deepseek", default="output/eval/deepseek_full5718.json")
    ap.add_argument("--gpt", default="output/eval/gpt-5.5_full5718.json")
    ap.add_argument("--eval-data", default="data/tom/tombench_eval.jsonl")
    ap.add_argument("--target-acc", type=float, default=0.8080,
                    help="Reference target (deepseek-v4-pro) for gap analysis")
    args = ap.parse_args()

    files = {
        'stage8_direct':  (args.stage11a, 'direct'),
        'stage8_cot':     (args.stage11a, 'cot'),
        'stage8_del_tom': (args.stage11a, 'del_tom'),
        'deepseek':       (args.deepseek, 'direct'),
        'gpt-5.5':        (args.gpt,      'direct'),
    }

    results = {}
    for name, (path, proto) in files.items():
        recs = json.load(open(path))
        sub = [r for r in recs if r.get('protocol') == proto]
        results[name] = {r['question_id']: r for r in sub}

    # Headline
    print(f'{"Model":<22} {"Acc":<10} {"vs s8 direct":<15}')
    print('-' * 50)
    s8_direct_qids = list(results['stage8_direct'].keys())
    s8_direct = sum(r['correct'] for r in results['stage8_direct'].values()) / len(s8_direct_qids)
    for name, idx in results.items():
        acc = sum(r['correct'] for r in idx.values()) / len(idx)
        delta = (acc - s8_direct) * 100
        print(f'{name:<22} {acc:.4f}    {delta:+.2f}pp')

    print()
    print('=== Per-task accuracy ===')
    print(f'{"Task":<18} {"s8_dir":<8} {"s8_cot":<8} {"s8_del":<8} {"deepseek":<10} {"gpt-5.5":<8}')
    print('-' * 70)
    ev_recs = {r['question_id']: r for r in [json.loads(l) for l in open(args.eval_data)]}
    tasks = defaultdict(list)
    for q in s8_direct_qids:
        tasks[ev_recs[q]['task']].append(q)

    for task in sorted(tasks):
        qids = tasks[task]
        accs = []
        for name in ['stage8_direct', 'stage8_cot', 'stage8_del_tom', 'deepseek', 'gpt-5.5']:
            n = sum(1 for q in qids if results[name][q]['correct'])
            accs.append(n / len(qids))
        print(f'{task:<18} {accs[0]:.4f}  {accs[1]:.4f}  {accs[2]:.4f}  {accs[3]:.4f}    {accs[4]:.4f}')

    print()
    print(f'=== Where s8 del_tom EXCEEDS deepseek ({args.target_acc}) ===')
    for task in sorted(tasks):
        qids = tasks[task]
        s8_del = sum(1 for q in qids if results['stage8_del_tom'][q]['correct']) / len(qids)
        ds = sum(1 for q in qids if results['deepseek'][q]['correct']) / len(qids)
        if s8_del > ds:
            print(f'  {task}: s8_del={s8_del:.4f} > deepseek={ds:.4f} (+{(s8_del-ds)*100:.2f}pp)')

    print()
    print('=== Where s8 del_tom STILL TRAILS deepseek (Stage 12 target gaps) ===')
    for task in sorted(tasks, key=lambda t: -(sum(1 for q in tasks[t] if results['deepseek'][q]['correct']) / len(tasks[t]) - sum(1 for q in tasks[t] if results['stage8_del_tom'][q]['correct']) / len(tasks[t]))):
        qids = tasks[task]
        s8_del = sum(1 for q in qids if results['stage8_del_tom'][q]['correct']) / len(qids)
        ds = sum(1 for q in qids if results['deepseek'][q]['correct']) / len(qids)
        if s8_del < ds:
            print(f'  {task}: s8_del={s8_del:.4f} < deepseek={ds:.4f} ({(s8_del-ds)*100:+.2f}pp gap, n={len(qids)})')


if __name__ == "__main__":
    main()
