"""Synth EmoBench-style + SocialIQA-style training data via DeepSeek-v4-pro.

Targets the 3 sub-tasks identified as v3.1 weakness:
  EU_emotion (6-opt emotion naming): 500 records
  EU_cause (4-opt cause-of-emotion): 300 records
  EA (4-opt action-choice): 300 records
  SocialIQA (3-opt commonsense): 400 records

Output schema matches data/tom/tom_train_stage14_weighted.jsonl format
(messages, ground_truth, tag, source, language, task, question_id).

Usage:
  python scripts/data/synth_emobench_socialiqa.py \
      --target eu_emotion --n 500 --out data/tom/raw/synth_eu_emotion.jsonl
  python scripts/data/synth_emobench_socialiqa.py \
      --target eu_cause --n 300 --out data/tom/raw/synth_eu_cause.jsonl
  python scripts/data/synth_emobench_socialiqa.py \
      --target ea --n 300 --out data/tom/raw/synth_ea.jsonl
  python scripts/data/synth_emobench_socialiqa.py \
      --target socialiqa --n 400 --out data/tom/raw/synth_socialiqa.jsonl

Run all 4 in parallel via shell `&`.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.clients import BackendSpec, ChatClient


# 6 emotion options (matched to EmoBench EU_emotion style)
EMOTIONS = [
    "Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust",
    "Anxiety", "Shame", "Guilt", "Pride", "Embarrassment", "Confusion",
    "Frustration", "Loneliness", "Jealousy", "Gratitude", "Hopefulness", "Resentment",
]


SYS_EU_EMOTION = """You are a Theory of Mind / emotion-recognition question generator.
Generate ONE training example for emotion-naming (6-option MCQ).

Output strict JSON with keys:
  scenario: 1-3 sentences describing a social situation a subject is in
  subject:  the name of the person whose emotion is being asked about
  question: ALWAYS "What is the most likely emotion of {subject}?"
  options:  list of EXACTLY 6 emotion words; the correct one + 5 plausible distractors
  gold:     0-indexed integer of the correct option

Rules:
- The scenario must clearly imply ONE specific emotion among the 6 options
- Distractors should be plausible but NOT equally fitting
- Pool of emotions to draw from: Happiness, Sadness, Anger, Fear, Surprise, Disgust, Anxiety, Shame, Guilt, Pride, Embarrassment, Confusion, Frustration, Loneliness, Jealousy, Gratitude, Hopefulness, Resentment
- Vary subject names (Alex, Maya, Jordan, Sara, etc.) and scenarios

Output JSON only, no markdown, no extra text."""


SYS_EU_CAUSE = """You are a Theory of Mind / emotion-cause question generator.
Generate ONE training example for cause-of-emotion (4-option MCQ).

Output strict JSON with keys:
  scenario: 2-4 sentences where the subject visibly experiences an emotion
  subject:  the name of the person whose emotion-cause is being asked about
  question: ALWAYS "What is the most likely cause of {subject}'s emotion?"
  options:  list of EXACTLY 4 cause statements; the correct one + 3 plausible distractors
  gold:     0-indexed integer of the correct option

Rules:
- Scenario must establish enough context to identify the cause
- Distractors should be related but ruled out by the scenario
- Vary subject names and emotional contexts

Output JSON only."""


SYS_EA = """You are a Theory of Mind / appropriate-action question generator.
Generate ONE training example for socially-appropriate action choice (4-option MCQ).

Output strict JSON with keys:
  scenario: 2-4 sentences describing a social/interpersonal dilemma a subject faces
  subject:  the person who needs to act
  question: ALWAYS "What is the most appropriate response or action for {subject}?"
  options:  list of EXACTLY 4 action choices; the correct one + 3 plausible-but-suboptimal alternatives
  gold:     0-indexed integer of the correct option

Rules:
- The "correct" option should be socially appropriate, balancing competing concerns
- Distractors should be tempting but flawed (too aggressive, too passive, ignores context, etc.)

Output JSON only."""


SYS_SOCIALIQA = """You are a SocialIQA-style commonsense reasoning question generator.
Generate ONE training example (3-option MCQ).

Output strict JSON with keys:
  context:   1-2 sentences describing what someone did/will do
  question:  a wh-question about wanting, feeling, needing, or what others will think (e.g. "How would others feel?", "What does X want to do next?", "Why did X do this?")
  options:   list of EXACTLY 3 short answer phrases; the correct one + 2 plausible distractors
  gold:      0-indexed integer of the correct option

Rules:
- Scenario must require everyday social commonsense, not formal logic
- Distractors should sound plausible but be slightly off (wrong actor, wrong tense, wrong intensity)
- Style: SocialIQA dev examples ("Tracy didn't go home that evening and resisted Riley's attacks.", "How would Tracy feel afterwards?")

Output JSON only."""


PROMPT_USER = "Generate ONE training example. Return JSON only."


TARGETS = {
    "eu_emotion": {
        "system": SYS_EU_EMOTION,
        "n_options": 6,
        "task": "EU_emotion",
        "source": "synth_emobench_eu_emotion",
        "qid_prefix": "synth_eu_emotion",
    },
    "eu_cause": {
        "system": SYS_EU_CAUSE,
        "n_options": 4,
        "task": "EU_cause",
        "source": "synth_emobench_eu_cause",
        "qid_prefix": "synth_eu_cause",
    },
    "ea": {
        "system": SYS_EA,
        "n_options": 4,
        "task": "EA",
        "source": "synth_emobench_ea",
        "qid_prefix": "synth_ea",
    },
    "socialiqa": {
        "system": SYS_SOCIALIQA,
        "n_options": 3,
        "task": "social_iqa",
        "source": "synth_socialiqa",
        "qid_prefix": "synth_socialiqa",
    },
}


def build_messages(target_cfg: dict) -> list[dict]:
    return [
        {"role": "system", "content": target_cfg["system"]},
        {"role": "user", "content": PROMPT_USER},
    ]


def parse_json_loose(s: str) -> dict | None:
    s = s.strip()
    # strip markdown fences
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # find first { ... } block
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def to_train_record(parsed: dict, target_cfg: dict, idx: int) -> dict | None:
    """Convert parsed synth output to standard training record."""
    n = target_cfg["n_options"]
    task = target_cfg["task"]

    if task == "social_iqa":
        story = parsed.get("context", "").strip()
        question = parsed.get("question", "").strip()
    else:
        story = parsed.get("scenario", "").strip()
        question = parsed.get("question", "").strip()
        subject = parsed.get("subject", "").strip()
        if subject and "{subject}" in question:
            question = question.replace("{subject}", subject)
        elif subject and "{" not in question and " " in question:
            # leave as-is; question template should already inline subject
            pass

    options = parsed.get("options", [])
    gold = parsed.get("gold", None)
    if not story or not question or not isinstance(options, list) or len(options) != n:
        return None
    if not isinstance(gold, int) or not (0 <= gold < n):
        return None
    options = [str(o).strip() for o in options]
    if any(not o for o in options):
        return None

    letter = chr(ord("A") + gold)
    letters_str = ", ".join(chr(ord("A") + i) for i in range(n))
    sys_p = (
        "You are a careful reader answering a multiple-choice question. "
        "Read the story and the question carefully, then output ONLY your final answer "
        f"in the format \\boxed{{X}} where X is one of {letters_str}. "
        "Do not include any explanation, reasoning, or extra text."
    )
    opts_block = "\n".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
    user_p = f"Story:\n{story}\n\nQuestion: {question}\n{opts_block}"

    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
        ],
        "ground_truth": letter,
        "tag": "tom_mcq",
        "source": target_cfg["source"],
        "language": "en",
        "task": task,
        "question_id": f"{target_cfg['qid_prefix']}_{idx:05d}",
    }


def synth_one(client: ChatClient, target_cfg: dict, idx: int) -> dict | None:
    msgs = build_messages(target_cfg)
    try:
        result = client.chat(
            messages=msgs,
            temperature=0.95,
            top_p=0.95,
            max_tokens=2048,
        )
    except Exception:
        return None
    content = result.content if hasattr(result, "content") else (result or "")
    parsed = parse_json_loose(content)
    if not parsed:
        return None
    return to_train_record(parsed, target_cfg, idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=list(TARGETS.keys()), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="deepseek-v4-pro")
    args = p.parse_args()

    target_cfg = TARGETS[args.target]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    spec = BackendSpec(name="deepseek", model=args.model)
    client = ChatClient(spec=spec)

    # Submit ~1.4× extra to allow for parse failures
    n_to_submit = int(args.n * 1.4)

    written = []
    fail = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(synth_one, client, target_cfg, i)
            for i in range(n_to_submit)
        ]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec:
                written.append(rec)
                if len(written) >= args.n:
                    break
            else:
                fail += 1

    written = written[: args.n]
    with open(args.out, "w") as f:
        for r in written:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(written)} records (failed {fail}) → {args.out}")


if __name__ == "__main__":
    main()
