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


def detect_question_type(question, expected_answer):
    """Classify question into yes_no, knowledge, or container.

    Returns (qtype, true_distractor_opt_strings) where the distractor strings
    are the *opposite* answer for yes/no and knowledge, and None for container
    (which uses smart object extraction).
    """
    q_lower = question.lower()
    ans_lower = expected_answer.lower().strip()

    if 'yes or no' in q_lower or ans_lower in ('yes', 'no'):
        # Yes/no question
        return ('yes_no', None)
    if ('knows about' in q_lower or 'know about' in q_lower
            or 'know it' in q_lower
            or 'know about it' in ans_lower
            or 'know it' in ans_lower):
        # Knowledge question
        return ('knowledge', None)
    # Default: container question
    return ('container', None)


def yes_no_distractors(gold):
    """For yes/no questions, the distractor is the opposite plus 2 plausible
    'I don't know' style options."""
    if gold.lower().strip() == 'yes':
        opp = 'no'
    else:
        opp = 'yes'
    return [opp, "I don't know", "It's unclear from the story"]


def knowledge_distractors(gold):
    """For knowledge questions, distractor is the negation plus 2 hedges."""
    g = gold.lower().strip()
    if 'does not' in g or "doesn't" in g or 'not know' in g:
        opp = 'knows about it'
    else:
        opp = 'does not know about it'
    return [opp, "I don't know", "It depends on context"]


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

# Filler words that signal an extracted phrase is not a clean noun-phrase
_BAD_PREFIXES = (
    "and ", "or ", "the ", "a ", "an ", "then ", "first ", "next ",
    "into ", "onto ", "after ", "before ", "behind ", "beside ",
    "near ", "from ", "with ", "without ", "as if ",
)


def _is_clean_phrase(p):
    """Reject phrases that start with conjunctions, adverbs, or prepositions
    that indicate the regex captured mid-sentence text rather than a real
    noun-phrase."""
    p_low = p.lower().strip()
    for prefix in _BAD_PREFIXES:
        if p_low.startswith(prefix) and prefix != "the " and prefix != "a " and prefix != "an ":
            # Articles are fine; conjunctions/adverbs/prepositions are not
            return False
    if p_low in _BAD_PREFIXES:
        return False
    # Must contain at least one of the container keywords as a whole token
    tokens = set(p_low.split())
    if not any(any(kw in t for kw in _CONTAINER_KEYWORDS.split('|')) for t in tokens):
        return False
    return True


def _normalize_for_match(s):
    """Strip articles and whitespace to detect near-duplicates."""
    s = s.lower().strip()
    for article in ['the ', 'a ', 'an ']:
        if s.startswith(article):
            s = s[len(article):]
    return s.strip()


def container_distractors(story, gold, num_candidates=3):
    """Extract plausible distractors from the story text.

    Filters out distractors that are near-duplicates of the gold answer
    (differing only by leading article, trailing punctuation, or substring).
    """
    gold_norm = _normalize_for_match(gold)
    phrases = _RE_PHRASES.findall(story.lower())

    candidates = []
    seen_norm = {gold_norm}
    for p in phrases:
        p = p.strip()
        if not p:
            continue
        if not _is_clean_phrase(p):
            continue
        p_norm = _normalize_for_match(p)
        # Reject if it normalizes to the gold, contains gold, or gold contains it
        if p_norm in seen_norm:
            continue
        if p_norm in gold_norm or gold_norm in p_norm:
            continue
        seen_norm.add(p_norm)
        candidates.append(p)

    if len(candidates) < num_candidates:
        fallbacks = [
            "kitchen counter", "wooden cabinet", "leather satchel",
            "metal drawer", "garden shed", "office desk",
            "storage closet", "back room", "hallway closet",
            "filing cabinet", "wall locker", "ceramic vase", "plastic crate",
        ]
        for f in fallbacks:
            f_norm = _normalize_for_match(f)
            if f_norm in seen_norm:
                continue
            if f_norm in gold_norm or gold_norm in f_norm:
                continue
            seen_norm.add(f_norm)
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

    qtype, _ = detect_question_type(question, gold_text)
    if qtype == 'yes_no':
        distractors = yes_no_distractors(gold_text)
    elif qtype == 'knowledge':
        distractors = knowledge_distractors(gold_text)
    else:
        distractors = container_distractors(story, gold_text, 3)

    if len(distractors) < 3:
        return None

    options = list(distractors[:3]) + [gold_text]
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
            "qtype": qtype,
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
