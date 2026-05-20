"""
Leakage check for Stage 11 new training data (Track B ExploreToM + Track C HOT synth)
against the eval set tombench_eval.jsonl.

Uses MinHash LSH + length-aware Jaccard for story+question fields.
Outputs both summary (any new data leaking?) and per-record drop list if
threshold > 0.85 similarity.
"""
import argparse
import json
import sys
from pathlib import Path
import re

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("pip install datasketch", file=sys.stderr); sys.exit(1)


def shingles(text: str, k: int = 5):
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return {text[i:i+k] for i in range(max(1, len(text) - k + 1))}


def make_mh(text, num_perm=128):
    mh = MinHash(num_perm=num_perm)
    for s in shingles(text):
        mh.update(s.encode('utf8'))
    return mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--new-data", nargs="+", required=True,
                    help="JSONL file(s) of new training data to check")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--output-suspect", default="output/analysis/leakage_suspects.jsonl")
    args = ap.parse_args()

    eval_recs = [json.loads(l) for l in open(args.eval)]
    print(f"Loaded {len(eval_recs)} eval records")

    lsh = MinHashLSH(threshold=args.threshold, num_perm=128)
    eval_idx = {}
    for i, r in enumerate(eval_recs):
        text = (r.get('story', '') + ' ' + r.get('question', '')).strip()
        if not text: continue
        mh = make_mh(text)
        eval_idx[r['question_id']] = (r, mh)
        lsh.insert(r['question_id'], mh)

    suspects = []
    n_total = 0
    for path in args.new_data:
        for line in open(path):
            r = json.loads(line)
            n_total += 1
            text = (r.get('story', '') + ' ' + r.get('question', '')).strip()
            if not text: continue
            mh = make_mh(text)
            hits = lsh.query(mh)
            if hits:
                eval_qid = hits[0]
                eval_r, eval_mh = eval_idx[eval_qid]
                jacc = mh.jaccard(eval_mh)
                if jacc >= args.threshold:
                    suspects.append({
                        'new_qid': r.get('question_id'),
                        'eval_qid': eval_qid,
                        'jaccard': jacc,
                        'new_story': r.get('story', '')[:100],
                        'eval_story': eval_r.get('story', '')[:100],
                    })

    print(f"\nChecked {n_total} new records vs {len(eval_recs)} eval records")
    print(f"Suspects (jaccard >= {args.threshold}): {len(suspects)}")
    if suspects:
        Path(args.output_suspect).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_suspect, 'w') as f:
            for s in suspects: f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"Wrote suspects to {args.output_suspect}")
        print("Top 5 suspect examples:")
        for s in sorted(suspects, key=lambda x: -x['jaccard'])[:5]:
            print(f"  jacc={s['jaccard']:.3f} new={s['new_qid']} eval={s['eval_qid']}")
    else:
        print("✅ No leakage detected.")


if __name__ == "__main__":
    main()
