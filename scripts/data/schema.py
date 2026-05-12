"""Unified data record schema (training + eval)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


# ToMBench's 20 ability strings → 8 broad task categories
ABILITY_TO_TASK = {
    "Belief: Location false beliefs": "False Belief",
    "Belief: Identity false beliefs": "False Belief",
    "Belief: Strange Story Task": "Strange Story",
    "Belief: Ambiguous Story": "Strange Story",
    "Belief: Unexpected Outcome": "Unexpected Outcome",
    "Belief: Persuasion Story": "Persuasion Story",
    "Belief: Knowledge-Attention Link": "Knowledge",
    "Belief: Knowledge-Pretend Play Link": "Knowledge",
    "Belief: Percepts-Knowledge Link": "Knowledge",
    "Desire: Multiple Desires": "Desire",
    "Desire: Discrepant Desires": "Desire",
    "Emotion: Moral Emotions": "Emotion",
    "Emotion: Discrepant Emotions": "Emotion",
    "Emotion: Hidden Emotions": "Emotion",
    "Emotion: Emotion Regulation": "Emotion",
    "Intention: Prediction of Actions": "Intention",
    "Intention: Discrepant Intentions": "Intention",
    "Intention: Completion of Failed Actions": "Intention",
    "Non-literal Comm: Hinting": "Non-literal Comm",
    "Non-literal Comm: Faux-pas Recognition": "Non-literal Comm",
    "Non-literal Comm: Scalar Implicature": "Non-literal Comm",
}


@dataclass
class TomRecord:
    """One record (eval or train) in unified schema."""
    question_id: str
    source: str                 # tombench | hi_tom | exploretom | simpletom | socialiqa | synth
    language: str               # en | zh
    task: str                   # one of the 8 ToMBench broad categories
    story: str
    question: str
    opt_a: str
    opt_b: str
    opt_c: str
    opt_d: str
    gold: str                   # "A" | "B" | "C" | "D"

    def to_jsonl_dict(self) -> dict:
        return asdict(self)


def ability_to_task(ability: str) -> str:
    """Map ToMBench 'ability' field to one of 8 broad tasks; default to 'Other'."""
    return ABILITY_TO_TASK.get(ability.strip(), "Other")
