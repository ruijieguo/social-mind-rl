"""Unified data record schema (training + eval)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


# Map ToMBench's verbose ability strings into the 6 broad task categories used
# by both the eval reports and the design spec. We match on prefix because the
# ability strings have many fine-grained variants (e.g.
# "Belief: Location false beliefs", "Belief: Sequence false beliefs", etc.).
TASK_PREFIXES = [
    ("Belief", "Belief"),
    ("Desire", "Desire"),
    ("Emotion", "Emotion"),
    ("Intention", "Intention"),
    ("Knowledge", "Knowledge"),
    ("Non-Literal Communication", "Non-literal Comm"),
    ("Non-literal communication", "Non-literal Comm"),
    ("Non-literal Comm", "Non-literal Comm"),
]

# Special-case mapping for ability strings that should be put in a different
# task than their literal prefix (e.g. Social-R1 reports false-belief explicitly).
ABILITY_TO_TASK = {
    "Belief: Location false beliefs": "False Belief",
    "Belief: Content false beliefs": "False Belief",
    "Belief: Sequence false beliefs": "False Belief",
    "Belief: Identity false beliefs": "False Belief",
    "Belief: Location false beliefs Belief: Second-order beliefs": "False Belief",
    "Belief: Content false beliefs Belief: Second-order beliefs": "False Belief",
}


@dataclass
class TomRecord:
    """One record (eval or train) in unified schema."""
    question_id: str
    source: str                 # tombench | hi_tom | exploretom | simpletom | socialiqa | synth
    language: str               # en | zh
    task: str                   # one of the broad ToMBench categories
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
    """Map ToMBench 'ability' field to a broad task category.

    Resolution order:
    1. Exact match in ABILITY_TO_TASK (false-belief subtypes).
    2. Prefix match in TASK_PREFIXES.
    3. Default to "Other".
    """
    s = ability.strip()
    if s in ABILITY_TO_TASK:
        return ABILITY_TO_TASK[s]
    for prefix, task in TASK_PREFIXES:
        if s.startswith(prefix + ":") or s.startswith(prefix + " "):
            return task
    return "Other"

