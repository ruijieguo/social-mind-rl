"""Deep error analysis: for our best model (14B-tom), classify errors by:
  - Task type
  - EN vs ZH
  - Whether deepseek-v4-pro also got it wrong (hard-ceiling vs catchable)
  - Predicted letter pattern (mode-collapse signs)
  - Story length / response length

Outputs:
  - output/analysis/14b_errors_categorized.json
    (one entry per error, with all metadata for GPT-5.5 audit)
  - output/analysis/14b_error_stats.md
    (summary tables for the report)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def main():
    s14 = load("output/eval/14b_full5718.json")
    ds = load("output/eval/deepseek_full5718.json")
    eval_prompts = {}
    with open("data/tom/tombench_eval.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qid = r.get("question_id") or r.get("ground_truth", {}).get("question_id")
            eval_prompts[qid] = r

    # filter to direct
    s14_idx = {r["question_id"]: r for r in s14 if r["model"] == "qwen3-14b-tom" and r["protocol"] == "direct"}
    ds_idx = {r["question_id"]: r for r in ds if r["model"] == "deepseek-v4-pro" and r["protocol"] == "direct"}

    errors = []
    for qid, r in s14_idx.items():
        if r["correct"]:
            continue
        ds_r = ds_idx.get(qid, {})
        prompt = eval_prompts.get(qid, {})
        story = prompt.get("messages", [{}])[1].get("content", "") if prompt.get("messages") else ""
        # extract just the story portion (before "Question:")
        story_part = story.split("Question:")[0].split("问题：")[0].strip() if story else ""
        errors.append({
            "qid": qid,
            "lang": r["language"],
            "task": r["task"],
            "gold": r["gold"],
            "our_pred": r["pred"],
            "ds_correct": ds_r.get("correct"),
            "ds_pred": ds_r.get("pred"),
            "story": story_part,
            "user_prompt": story,
            "our_response": (r.get("raw_responses") or [""])[0][:2000],
            "ds_response": (ds_r.get("raw_responses") or [""])[0][:1500] if ds_r else "",
        })

    print(f"Total 14B errors: {len(errors)} out of 5718 ({len(errors)*100/5718:.1f}%)")
    print()

    # by task
    by_task = Counter(e["task"] for e in errors)
    by_task_lang = defaultdict(Counter)
    for e in errors:
        by_task_lang[e["task"]][e["lang"]] += 1
    print("Errors by task:")
    for t, n in sorted(by_task.items(), key=lambda x: -x[1]):
        en = by_task_lang[t]["en"]; zh = by_task_lang[t]["zh"]
        print(f"  {t:25s} n={n:4d}  EN={en:4d}  ZH={zh:4d}")
    print()

    # ds also wrong (hard ceiling)
    both_wrong = sum(1 for e in errors if not e["ds_correct"])
    only_we_wrong = sum(1 for e in errors if e["ds_correct"])
    print(f"Errors where deepseek was ALSO wrong (hard ceiling / label issue?):  {both_wrong}  ({both_wrong*100/len(errors):.1f}% of errors)")
    print(f"Errors where deepseek was RIGHT (we could catch up):                 {only_we_wrong}  ({only_we_wrong*100/len(errors):.1f}% of errors)")
    print()

    # by task: hard ceiling rate
    print("Hard-ceiling errors per task (% of task's errors where deepseek was also wrong):")
    for t, n in sorted(by_task.items(), key=lambda x: -x[1]):
        n_both = sum(1 for e in errors if e["task"] == t and not e["ds_correct"])
        rate = n_both * 100 / n if n else 0
        print(f"  {t:25s}  both_wrong={n_both:3d}/{n:3d}  rate={rate:.1f}%")
    print()

    # confusion matrix (gold -> our_pred)
    print("Confusion (gold -> our_pred):")
    cm = defaultdict(Counter)
    for e in errors:
        cm[e["gold"]][e["our_pred"]] += 1
    print("  gold | A   B   C   D")
    for g in "ABCD":
        row = [cm[g][p] for p in "ABCD"]
        print(f"   {g}  | " + " ".join(f"{x:3d}" for x in row))
    print()

    # save full error list as JSON
    out_path = Path("output/analysis/14b_errors_categorized.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2))
    print(f"wrote {len(errors)} error records to {out_path}")


if __name__ == "__main__":
    main()
