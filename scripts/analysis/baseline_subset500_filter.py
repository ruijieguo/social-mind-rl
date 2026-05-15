"""Filter baseline_combined.json to the 500 qids in tombench_eval_subset500.jsonl
then aggregate per (model, protocol) for an apples-to-apples comparison with
final_subset500.json."""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("output/eval/baseline_subset500.json")
SUB_PATH = Path("data/tom/tombench_eval_subset500.jsonl")
BASELINE = Path("output/eval/baseline_combined.json")

sub_qids = set()
with SUB_PATH.open() as f:
    for line in f:
        r = json.loads(line)
        qid = r.get("question_id") or r.get("ground_truth", {}).get("question_id")
        if qid:
            sub_qids.add(qid)

baseline = json.loads(BASELINE.read_text())
filtered = [r for r in baseline if r.get("question_id") in sub_qids]
OUT.write_text(json.dumps(filtered))
print(f"filtered {len(filtered)} of {len(baseline)} baseline records → {OUT}")

# aggregate
agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in filtered:
    model = r["model"]
    proto = r["protocol"]
    agg[(model, proto)]["all"][0] += int(r["correct"])
    agg[(model, proto)]["all"][1] += 1
    agg[(model, proto)][r["language"]][0] += int(r["correct"])
    agg[(model, proto)][r["language"]][1] += 1

print()
print("| Model | Protocol | n | Overall | EN | ZH |")
print("|---|---|---|---|---|---|")
for (m, p), buckets in sorted(agg.items()):
    n = buckets["all"][1]
    o = buckets["all"][0] / n if n else 0
    en_n = buckets["en"][1]; en = buckets["en"][0] / en_n if en_n else 0
    zh_n = buckets["zh"][1]; zh = buckets["zh"][0] / zh_n if zh_n else 0
    print(f"| {m} | {p} | {n} | {o:.4f} | {en:.4f} | {zh:.4f} |")
