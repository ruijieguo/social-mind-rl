#!/usr/bin/env bash
# Verify production_frozen/v3.1 artifacts haven't been corrupted.
# Usage: bash production_frozen/v3.1/verify.sh   (run from repo root)
set -e
cd "$(dirname "$0")/../.."   # go to repo root so SHA paths resolve relative to v3.1/

cd production_frozen/v3.1

echo "Verifying SHA-256 checksums (production_frozen/v3.1)..."
shasum -a 256 -c <(grep -E "^[0-9a-f]{64}" SHA256SUMS.txt) && echo "OK: All checksums match." || {
    echo "FAIL: Some files have been modified."
    exit 1
}

echo
echo "Record counts:"
echo "  tom_train_14b_stage14b_weighted.jsonl  expected 14408, actual $(wc -l < data/tom_train_14b_stage14b_weighted.jsonl)"
echo "  tom_train_14b_stage12.jsonl            expected 12519, actual $(wc -l < data/tom_train_14b_stage12.jsonl)"
echo "  exploretom_v2_track_b.jsonl            expected  2000, actual $(wc -l < data/raw/exploretom_v2_track_b.jsonl)"
echo "  synth_gpt55_phase_d_hot_track_c        expected  1260, actual $(wc -l < data/raw/synth_gpt55_phase_d_hot_track_c.jsonl)"

echo
echo "Headline result spot-check (Stage 14b ckpt-199 final, full 5718):"
python3 << 'PYEOF'
import json
recs = json.load(open('eval/14b_stage14b_ckpt199_full5718.json'))
for p in ('direct', 'cot', 'del_tom'):
    sub = [r for r in recs if r.get('protocol') == p]
    if sub:
        acc = sum(r['correct'] for r in sub) / len(sub)
        n = len(sub)
        expected = {'direct': 0.7721, 'cot': 0.7754, 'del_tom': 0.7875}[p]
        marker = '✓' if abs(acc - expected) < 0.001 else '✗'
        print(f'  {p:>8}: {acc:.4f}  (n={n}, expected {expected:.4f}) {marker}')

print()
print("Per-task del_tom (key gains):")
recs_dt = [r for r in recs if r.get('protocol') == 'del_tom']
from collections import defaultdict
by_t = defaultdict(lambda: {'c': 0, 'n': 0})
for r in recs_dt:
    t = r.get('task', '?')
    by_t[t]['n'] += 1
    by_t[t]['c'] += int(r['correct'])
for t in ('Knowledge', 'Belief', 'Emotion', 'False Belief'):
    if t in by_t:
        acc = by_t[t]['c'] / by_t[t]['n']
        print(f'  {t:>22}: {acc:.4f}  (n={by_t[t]["n"]})')
PYEOF

echo
echo "v3.1 verification complete."
