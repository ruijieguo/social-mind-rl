"""Prompt templates + answer extractors for the Qwen3-14B full-eval.

Self-contained (copied, not imported) from scripts/eval/run_tombench.py,
run_generic_mcq.py, extractors.py — so this experiment can be rsynced to the
GPU host and run without the rest of the repo.

Benchmarks & option counts:
  tombench   : 4 options (A-D), ZH/EN bilingual, ToM-specific system prompt
  hitom      : 15 options (A-O), EN-only, generic MCQ system prompt
  socialiqa  : 3 options (A-C),  EN-only, generic MCQ system prompt
  emobench   : 4 options (A-D),  EN-only, generic MCQ system prompt

The tombench prompts are kept verbatim from production_frozen so numbers stay
comparable to history; hitom/socialiqa/emobench all use the SAME generic MCQ
prompt builder that the unified eval (scripts/eval/run_generic_mcq.py) uses.

Sampling params (2026-05-29 — max_tokens raised to 8192 per user, except direct):
  direct        : T=0.0, max_tokens=64,    thinking=false, n=1
  direct_think  : T=0.0, max_tokens=8192,  thinking=true,  n=1
  cot           : T=0.6, max_tokens=8192,  thinking=true,  n=1
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# ToMBench (4 options A-D) — ToM-specific system prompt (verbatim from frozen)
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


_LETTER_PREFIX_PATTERN = re.compile(r"^\s*([A-D])[.．、:：]\s*", re.IGNORECASE)


def _strip_letter_prefix(opt: str) -> str:
    """Remove a leading 'A.' / 'A、' / 'A:' prefix that the dataset has baked into
    `opt_a/b/c/d` for ZH ToMBench records. EN records don't have it; ZH ones do.
    Calling this unconditionally is safe for EN (no match → no change)."""
    if opt is None:
        return ""
    m = _LETTER_PREFIX_PATTERN.match(opt)
    if m:
        return opt[m.end():]
    return opt


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
    # Strip baked-in 'A./B./C./D.' prefix from ZH options before formatting
    # (no-op for EN options that don't carry such a prefix).
    user = builder(
        story=record["story"], question=record["question"],
        opt_a=_strip_letter_prefix(record["opt_a"]),
        opt_b=_strip_letter_prefix(record["opt_b"]),
        opt_c=_strip_letter_prefix(record["opt_c"]),
        opt_d=_strip_letter_prefix(record["opt_d"]),
    )
    sys_p = SYSTEM_PROMPT_DIRECT_TOMBENCH if protocol in ("direct", "direct_think") else SYSTEM_PROMPT_COT_TOMBENCH
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Generic MCQ (Hi-ToM 15, SocialIQA 3, EmoBench 4) — option count is dynamic
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

    def has_boxed(text: str) -> bool:
        return bool(text) and bool(boxed_pattern.search(text))

    def vote(answers: Sequence[Optional[str]]) -> Optional[str]:
        valid = [a for a in answers if a in valid_letters]
        if not valid:
            return None
        counts = Counter(valid)
        max_count = max(counts.values())
        winners = sorted(c for c, n in counts.items() if n == max_count)
        return winners[0]

    return extract_direct, extract_cot, has_boxed, vote


# Pre-built extractor bundles per option count seen in this eval.
EXTRACT_BY_NOPTS = {
    3: make_extractors(3),    # SocialIQA
    4: make_extractors(4),    # ToMBench, EmoBench
    15: make_extractors(15),  # Hi-ToM
}


def extractors_for(num_options: int):
    if num_options not in EXTRACT_BY_NOPTS:
        EXTRACT_BY_NOPTS[num_options] = make_extractors(num_options)
    return EXTRACT_BY_NOPTS[num_options]


# ---------------------------------------------------------------------------
# Sampling parameters per protocol (2026-05-29)
# ---------------------------------------------------------------------------

def sampling_params_for(protocol: str) -> dict:
    if protocol == "direct":
        return dict(temperature=0.0, top_p=1.0, max_tokens=64,
                    n_samples=1, enable_thinking=False)
    if protocol == "direct_think":
        return dict(temperature=0.0, top_p=1.0, max_tokens=8192,
                    n_samples=1, enable_thinking=True)
    if protocol == "cot":
        return dict(temperature=0.6, top_p=0.95, max_tokens=8192,
                    n_samples=1, enable_thinking=True)
    raise ValueError(f"unknown protocol: {protocol}")
