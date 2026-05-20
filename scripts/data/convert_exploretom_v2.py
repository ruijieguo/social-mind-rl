"""
Track B: Convert Meta's official ExploreToM-data-sample.csv into ToMBench training format.

The CSV has 13309 records with fields:
  story_structure, infilled_story, question, expected_answer,
  qprop=params, qprop=nth_order, qprop=non_unique_mental_state,
  sprop=is_false_belief_story_1st, sprop=is_false_belief_story_1st_and_2nd,
  sprop=story_accuracy_1st_raw, sprop=story_accuracy_1st_infilled, ...

We convert each record into a 4-option MCQ.
Output: data/tom/raw/exploretom_v2.jsonl
"""
import argparse
import ast
import csv
import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_qprop_params(s):
    """qprop=params is a tuple-like string e.g. ('Liam', 'silver letter opener', 'memory-container_location')"""
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def task_label_from_props(rec):
    """Map ExploreToM properties to ToMBench task labels."""
    nth = rec.get("qprop=nth_order", "0")
    is_fb_1st = rec.get("sprop=is_false_belief_story_1st") == "TRUE"
    is_fb_2nd = rec.get("sprop=is_false_belief_story_1st_and_2nd") == "TRUE"

    if nth == "-1":
        return "Knowledge"  # memory/factual
    elif nth == "1":
        return "False Belief" if is_fb_1st else "Belief"
    elif nth == "2":
        return "False Belief" if is_fb_2nd else "Belief"
    return "Belief"


_CONTAINER_KEYWORDS = (
    "bag|box|drawer|briefcase|crate|cabinet|chest|cooler|jar|envelope|wallet|"
    "pocket|pouch|locker|safe|container|bin|tray|case|holder|bottle|tin|barrel|"
    "shelf|rack|vault|hat|notebook|folder|"
    "kitchen|lobby|hallway|garage|attic|basement|office|library|garden|patio|"
    "balcony|cellar|conference room|waiting room|living room|bedroom|bathroom|"
    "dining room|laundry room|playroom|study|den|porch|driveway|cafeteria|"
    "courtroom|warehouse|gymnasium|loft|hangar|barn"
)

_RE_PHRASES = re.compile(rf"\b(?:[a-z]+ ){{0,3}}(?:{_CONTAINER_KEYWORDS})\b")


def extract_candidates(story, gold, num_candidates=3):
    """Extract plausible distractors from the story text."""
    phrases = _RE_PHRASES.findall(story.lower())
    candidates = list({p.strip() for p in phrases if p and p.strip().lower() != gold.lower()})
    if len(candidates) < num_candidates:
        fallbacks = [
            "kitchen counter", "wooden cabinet", "leather satchel",
            "metal drawer", "garden shed", "office desk",
            "storage closet", "back room", "hallway closet",
        ]
        for f in fallbacks:
            if f.lower() != gold.lower() and f not in candidates:
                candidates.append(f)
            if len(candidates) >= num_candidates:
                break
    random.shuffle(candidates)
    return candidates[:num_candidates]


def build_record(idx, rec, rng):
    story = rec["infilled_story"].strip()
    question = rec["question"].strip()
    gold_text = rec["expected_answer"].strip()

    if not story or not question or not gold_text:
        return None
    if len(story) > 1500 or len(story) < 80:
        return None

    distractors = extract_candidates(story, gold_text, 3)
    if len(distractors) < 3:
        return None

    options = distractors + [gold_text]
    rng.shuffle(options)
    gold_letter = "ABCD"[options.index(gold_text)]

    task = task_label_from_props(rec)
    qid = f"exploretom_v2__{idx:06d}__nth{rec.get('qprop=nth_order','?')}"

    return {
        "question_id": qid,
        "source": "exploretom_v2",
        "language": "en",
        "task": task,
        "story": story,
        "question": question,
        "opt_a": options[0],
        "opt_b": options[1],
        "opt_c": options[2],
        "opt_d": options[3],
        "gold": gold_letter,
        "_meta": {
            "nth_order": rec.get("qprop=nth_order"),
            "is_fb_1st": rec.get("sprop=is_false_belief_story_1st"),
            "is_fb_2nd": rec.get("sprop=is_false_belief_story_1st_and_2nd"),
            "llama70b_acc_infilled": rec.get("sprop=story_accuracy_1st_infilled"),
            "story_type": rec.get("param=story_type"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/tom/raw/exploretom_v2_meta/data.csv")
    ap.add_argument("--output", default="data/tom/raw/exploretom_v2.jsonl")
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--min-llama-acc", type=float, default=0.0)
    ap.add_argument("--max-llama-acc", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = list(csv.DictReader(open(args.input)))
    print(f"Loaded {len(rows)} records from {args.input}")

    def get_acc(r):
        try:
            return float(r.get("sprop=story_accuracy_1st_infilled", -1))
        except Exception:
            return -1.0

    filtered = []
    for r in rows:
        a = get_acc(r)
        if 0 <= a <= args.max_llama_acc and a >= args.min_llama_acc:
            filtered.append(r)
    print(f"After Llama-70B acc filter [{args.min_llama_acc}, {args.max_llama_acc}]: {len(filtered)}")

    rng.shuffle(filtered)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    by_task = defaultdict(int)
    with open(args.output, "w") as out:
        for i, r in enumerate(filtered):
            if n_written >= args.max_records:
                break
            built = build_record(i, r, rng)
            if built is None:
                n_skipped += 1
                continue
            by_task[built["task"]] += 1
            out.write(json.dumps(built, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nWritten {n_written} records (skipped {n_skipped}) to {args.output}")
    print("By task:")
    for t, n in sorted(by_task.items(), key=lambda x: -x[1]):
        print(f"  {t:<20} {n}")


if __name__ == "__main__":
    main()
