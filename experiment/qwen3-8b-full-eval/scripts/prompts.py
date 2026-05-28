"""Prompt templates + answer extractors for the Qwen3-8B full-eval.

Copied (not imported) from scripts/eval/run_tombench.py, run_generic_mcq.py,
extractors.py, extractors_generic.py to keep this experiment self-contained
and avoid touching project-wide eval code.

Sampling params reflect the user's 2026-05-28 request:
  direct  : T=0.0, max_tokens=64,   thinking=false, n=1
  cot     : T=0.6, max_tokens=4096, thinking=true,  n=1
  del_tom : T=0.7, max_tokens=4096, thinking=true,  n=8 (majority vote)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# ToMBench (4 options A-D)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DIRECT_TOMBENCH = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Read the story and the question carefully, then output ONLY your final answer "
    "in the format \\boxed{X} where X is one of A, B, C, D. "
    "Do not include any explanation, reasoning, or extra text."
)

SYSTEM_PROMPT_COT_TOMBENCH = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Think step by step about the mental states of the characters, "
    "then output your final answer in the format \\boxed{X} where X is one of A, B, C, D. "
    "Put your final \\boxed{X} on the last line."
)


def _user_prompt_tombench_en(*, story, question, opt_a, opt_b, opt_c, opt_d) -> str:
    return (
        f"Story:\n{story}\n\n"
        f"Question: {question}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
    )


def _user_prompt_tombench_zh(*, story, question, opt_a, opt_b, opt_c, opt_d) -> str:
    return (
        f"故事：\n{story}\n\n"
        f"问题：{question}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
    )


def build_messages_tombench(record: dict, *, protocol: str) -> list[dict]:
    builder = _user_prompt_tombench_zh if record.get("language") == "zh" else _user_prompt_tombench_en
    user = builder(
        story=record["story"], question=record["question"],
        opt_a=record["opt_a"], opt_b=record["opt_b"],
        opt_c=record["opt_c"], opt_d=record["opt_d"],
    )
    # direct and direct_think both use the historical SYSTEM_PROMPT_DIRECT;
    # cot/del_tom use SYSTEM_PROMPT_COT.
    sys_p = SYSTEM_PROMPT_DIRECT_TOMBENCH if protocol in ("direct", "direct_think") else SYSTEM_PROMPT_COT_TOMBENCH
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Generic MCQ (Hi-ToM, A-R 18 options)
# ---------------------------------------------------------------------------

SYSTEM_DIRECT_TPL_GENERIC = (
    "You are a careful reader answering a multiple-choice question. "
    "Read the story (if any) and the question carefully, then output ONLY your final answer "
    "in the format \\boxed{{X}} where X is one of {letters}. "
    "Do not include any explanation, reasoning, or extra text."
)

SYSTEM_COT_TPL_GENERIC = (
    "You are a careful reader answering a multiple-choice question. "
    "Think step by step about the question, then output your final answer "
    "in the format \\boxed{{X}} where X is one of {letters}. "
    "Put your final \\boxed{{X}} on the last line."
)


def build_messages_generic(record: dict, *, protocol: str) -> list[dict]:
    options = record["options"]
    n = len(options)
    letters = ", ".join(chr(ord("A") + i) for i in range(n))
    sys_tpl = SYSTEM_DIRECT_TPL_GENERIC if protocol in ("direct", "direct_think") else SYSTEM_COT_TPL_GENERIC
    system = sys_tpl.format(letters=letters)

    opts_block = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options))
    story = record.get("story", "")
    if record.get("language") == "zh":
        user = (f"故事：\n{story}\n\n" if story else "") + f"问题：{record['question']}\n" + opts_block
    else:
        user = (f"Story:\n{story}\n\n" if story else "") + f"Question: {record['question']}\n" + opts_block
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def make_extractors(num_options: int):
    if num_options < 2 or num_options > 26:
        raise ValueError(f"num_options must be in [2, 26], got {num_options}")
    valid_letters = {chr(ord("A") + i) for i in range(num_options)}
    letter_class = "".join(sorted(valid_letters))
    boxed_pattern = re.compile(rf"\\boxed\{{([{letter_class}])\}}")

    def _first_letter(text: str) -> Optional[str]:
        for ch in text:
            if ch in valid_letters:
                return ch
        return None

    def _last_letter(text: str, tail_chars: int = 200) -> Optional[str]:
        tail = text[-tail_chars:]
        last = None
        for ch in tail:
            if ch in valid_letters:
                last = ch
        return last

    def extract_direct(text: str) -> Optional[str]:
        if not text:
            return None
        m = boxed_pattern.search(text)
        if m:
            return m.group(1)
        return _first_letter(text)

    def extract_cot(text: str) -> Optional[str]:
        if not text:
            return None
        matches = boxed_pattern.findall(text)
        if matches:
            return matches[-1]
        return _last_letter(text)

    def vote_del_tom(answers: Sequence[Optional[str]]) -> Optional[str]:
        valid = [a for a in answers if a in valid_letters]
        if not valid:
            return None
        counts = Counter(valid)
        max_count = max(counts.values())
        winners = sorted(c for c, n in counts.items() if n == max_count)
        return winners[0]

    return extract_direct, extract_cot, vote_del_tom


EXTRACT_TOMBENCH = make_extractors(4)   # A-D
EXTRACT_HITOM_18 = make_extractors(18)  # A-R


# ---------------------------------------------------------------------------
# Sampling parameters per protocol (user-specified 2026-05-28)
#
# direct_think is the historical-style direct (matches production_frozen/8b/v1.0
# 0.7450 setting): same SYSTEM_PROMPT_DIRECT, but thinking left at default-true
# and max_tokens 2048. Model usually ignores "output ONLY" and produces
# <think>...</think>\boxed{X}, so we use the cot-style extractor (last \boxed).
# ---------------------------------------------------------------------------

def sampling_params_for(protocol: str, n_samples_default: int = 8) -> dict:
    if protocol == "direct":
        return dict(temperature=0.0, top_p=1.0, max_tokens=64,
                    n_samples=1, enable_thinking=False)
    if protocol == "direct_think":
        return dict(temperature=0.0, top_p=1.0, max_tokens=2048,
                    n_samples=1, enable_thinking=True)
    if protocol == "cot":
        return dict(temperature=0.6, top_p=0.95, max_tokens=4096,
                    n_samples=1, enable_thinking=True)
    if protocol == "del_tom":
        return dict(temperature=0.7, top_p=0.95, max_tokens=4096,
                    n_samples=n_samples_default, enable_thinking=True)
    raise ValueError(f"unknown protocol: {protocol}")
