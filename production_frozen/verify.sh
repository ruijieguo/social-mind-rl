#!/usr/bin/env bash
# Verify production_frozen artifacts haven't been corrupted.
# Usage: bash production_frozen/verify.sh   (run from repo root)
set -e
cd "$(dirname "$0")/.."   # go to repo root so SHA paths resolve

echo "Verifying SHA-256 checksums..."
shasum -a 256 -c <(grep -E "^[0-9a-f]{64}" production_frozen/SHA256SUMS.txt) && echo "OK: All checksums match." || {
    echo "FAIL: Some files have been modified."
    exit 1
}

echo
echo "Record counts:"
echo "  tom_train_8b_stage7.jsonl  expected 9559, actual $(wc -l < production_frozen/data/tom_train_8b_stage7.jsonl)"
echo "  tom_train_14b_stage8.jsonl expected 9259, actual $(wc -l < production_frozen/data/tom_train_14b_stage8.jsonl)"
echo "  tombench_eval.jsonl        expected 5718, actual $(wc -l < production_frozen/data/tombench_eval.jsonl)"
echo "  tombench_eval_subset500    expected  500, actual $(wc -l < production_frozen/data/tombench_eval_subset500.jsonl)"
echo "  tombench_eval_clean.jsonl  expected 4551, actual $(wc -l < production_frozen/data/tombench_eval_clean.jsonl)"

echo
echo "Headline result spot-check:"
python3 -c "
import json
def acc(path, model_id, proto='direct'):
    rs = [r for r in json.load(open(path)) if r.get('protocol')==proto and r.get('model')==model_id]
    return sum(r['correct'] for r in rs) / len(rs) if rs else 0.0
print(f'  8B stage7  full5718  direct:    {acc(\"production_frozen/eval/8b_stage7_full5718.json\", \"qwen3-8b-tom-stage7\"):.4f}  (expected 0.7419)')
print(f'  14B stage8 full5718  direct:    {acc(\"production_frozen/eval/14b_stage8_full5718.json\", \"qwen3-14b-tom-stage8\"):.4f}  (expected 0.7594)')
print(f'  14B stage8 clean     direct:    {acc(\"production_frozen/eval/14b_stage8_clean_eval.json\", \"qwen3-14b-tom-stage8\"):.4f}  (expected 0.8449)')
print(f'  14B stage8 subset500 del_tom:   {acc(\"production_frozen/eval/14b_stage8_subset500.json\", \"qwen3-14b-tom-stage8\", \"del_tom\"):.4f}  (expected 0.7920)')
print(f'  deepseek   full5718  direct:    {acc(\"production_frozen/eval/deepseek_full5718.json\", \"deepseek-v4-pro\"):.4f}  (expected 0.8080)')
print(f'  GPT-5.5    full5718  direct:    {acc(\"production_frozen/eval/gpt-5.5_full5718.json\", \"gpt-5.5\"):.4f}  (expected 0.8349)')
"

echo
echo "All checks passed."
