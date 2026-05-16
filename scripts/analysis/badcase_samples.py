"""Sample concrete error cases for the bad-case analysis writeup.

Pull representative errors per task: closeable (deepseek correct, stage2 wrong)
and both-wrong (the hard-ceiling examples). Group by language and dump the
full story + question + options + gold + stage2 response.

Source data: tombench_eval.jsonl has the prompts; result JSONs have
the model's raw_responses.
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def load_jsonl(p):
    out = []
    with Path(p).open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def index_results(records, model_filter=None, proto="direct"):
    out = {}
    for r in records:
        if model_filter and r["model"] != model_filter:
            continue
        if r["protocol"] != proto:
            continue
        out[r["question_id"]] = r
    return out


def main():
    random.seed(7)
    s2 = load("output/eval/stage2_full5718.json")
    base = load("output/eval/baseline_combined.json")
    prompts_jsonl = load_jsonl("data/tom/tombench_eval.jsonl")
    qid_to_prompt = {}
    for r in prompts_jsonl:
        qid = r.get("question_id") or r.get("ground_truth", {}).get("question_id")
        if qid:
            qid_to_prompt[qid] = r

    s2_idx = index_results(s2, "qwen3-8b-tom-stage2")
    ds_idx = index_results(base, "deepseek-v4-pro")

    closeable = defaultdict(list)
    both_wrong = defaultdict(list)
    for qid, d in ds_idx.items():
        r2 = s2_idx.get(qid)
        if not r2:
            continue
        if d["correct"] and not r2["correct"]:
            closeable[r2["task"]].append(qid)
        elif not d["correct"] and not r2["correct"]:
            both_wrong[r2["task"]].append(qid)

    print("# Bad-case samples - concrete error analysis")
    print()
    print("Sample 3 closeable + 2 both-wrong per task. Closeable = deepseek")
    print("answered correctly, our stage2 model did not. Both-wrong = neither")
    print("got it (likely label noise or genuine ambiguity).")
    print()
    for task in sorted(closeable):
        print(f"## Task: {task}")
        print()
        print(f"### Closeable - sample of {min(3, len(closeable[task]))} of {len(closeable[task])}")
        print()
        sample_qids = random.sample(closeable[task], min(3, len(closeable[task])))
        for qid in sample_qids:
            prompt = qid_to_prompt.get(qid)
            r2 = s2_idx[qid]
            dump_case(qid, prompt, r2, ds_idx[qid], label="closeable")
        if both_wrong[task]:
            print(f"### Both wrong - sample of {min(2, len(both_wrong[task]))} of {len(both_wrong[task])}")
            print()
            sample_qids = random.sample(both_wrong[task], min(2, len(both_wrong[task])))
            for qid in sample_qids:
                prompt = qid_to_prompt.get(qid)
                r2 = s2_idx[qid]
                dump_case(qid, prompt, r2, ds_idx[qid], label="both-wrong")


def dump_case(qid, prompt, s2_record, ds_record, label):
    if prompt is None:
        return
    story = (prompt.get("messages") or [{}])
    user_msg = next((m["content"] for m in story if m.get("role") == "user"), "")
    raw_s2 = (s2_record.get("raw_responses") or [""])[0]
    raw_ds = (ds_record.get("raw_responses") or [""])[0]

    print(f"#### {qid} (lang={s2_record['language']}, gold={s2_record['gold']})")
    print()
    user_short = user_msg[:1200] + ("..." if len(user_msg) > 1200 else "")
    print("```")
    print(user_short.strip())
    print("```")
    print()
    print(f"  - **gold**: {s2_record['gold']} | **stage2 pred**: {s2_record['pred']} | **deepseek pred**: {ds_record['pred']}")
    print()
    s2_short = raw_s2[:900] + ("..." if len(raw_s2) > 900 else "")
    print(f"  stage2 response:")
    print("  ```")
    for line in s2_short.strip().splitlines():
        print(f"  {line}")
    print("  ```")
    print()
    if label == "closeable":
        ds_short = raw_ds[:600] + ("..." if len(raw_ds) > 600 else "")
        print(f"  deepseek response:")
        print("  ```")
        for line in ds_short.strip().splitlines():
            print(f"  {line}")
        print("  ```")
        print()


if __name__ == "__main__":
    main()
