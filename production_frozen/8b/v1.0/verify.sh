#!/usr/bin/env bash
# Verify production_frozen/8b/v1.0 artifacts haven't been corrupted.
# Usage: bash production_frozen/8b/v1.0/verify.sh   (run from repo root)
set -e
cd "$(dirname "$0")/../../.."   # go to repo root so SHA paths resolve relative to v1.0/

cd production_frozen/8b/v1.0

echo "Verifying SHA-256 checksums (production_frozen/8b/v1.0)..."
shasum -a 256 -c <(grep -E "^[0-9a-f]{64}" SHA256SUMS.txt) && echo "OK: All checksums match." || {
    echo "FAIL: Some files have been modified."
    exit 1
}

echo
echo "Record counts:"
echo "  tom_train_stage15_8b_filtered_weighted.jsonl  expected 7482, actual $(wc -l < data/tom_train_stage15_8b_filtered_weighted.jsonl)"
echo "  tom_train_14b_stage12.jsonl                   expected 12519, actual $(wc -l < data/tom_train_14b_stage12.jsonl)"
echo "  8b_stage7_reward_full12519.jsonl              expected 12519, actual $(wc -l < data/8b_stage7_reward_full12519.jsonl)"
echo "  exploretom_v2_track_b.jsonl                   expected  2000, actual $(wc -l < data/raw/exploretom_v2_track_b.jsonl)"
echo "  synth_gpt55_phase_d_hot_track_c               expected  1260, actual $(wc -l < data/raw/synth_gpt55_phase_d_hot_track_c.jsonl)"

echo
echo "Headline result spot-check (Stage 15 ckpt-150, full 5718):"
python3 << 'PYEOF'
import json
recs = json.load(open('eval/8b_stage15_ckpt150_full5718.json'))
for p in ('direct', 'cot', 'del_tom'):
    sub = [r for r in recs if r.get('protocol') == p]
    if sub:
        acc = sum(int(r.get('correct', False) if 'correct' in r else (r.get('pred') == r.get('gold'))) for r in sub) / len(sub)
        n = len(sub)
        expected = {'direct': 0.7450, 'cot': 0.7501, 'del_tom': 0.7618}[p]
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
    by_t[t]['c'] += int(r.get('correct', False) if 'correct' in r else (r.get('pred') == r.get('gold')))
for t in ('Knowledge', 'Belief', 'Intention', 'False Belief'):
    if t in by_t:
        acc = by_t[t]['c'] / by_t[t]['n']
        print(f'  {t:>22}: {acc:.4f}  (n={by_t[t]["n"]})')
PYEOF

echo
echo "8b/v1.0 verification complete."
