"""Answer extractors for the three evaluation protocols."""
import re
from collections import Counter
from typing import Optional, Sequence

_BOXED_PATTERN = re.compile(r"\\boxed\{([A-D])\}")
_VALID = {"A", "B", "C", "D"}


def _first_capital_letter(text: str) -> Optional[str]:
    """Return the first standalone A/B/C/D in text, else None."""
    for ch in text:
        if ch in _VALID:
            return ch
    return None


def _last_capital_letter(text: str, tail_chars: int = 200) -> Optional[str]:
    """Return the last A/B/C/D within the last `tail_chars` of text."""
    tail = text[-tail_chars:]
    last = None
    for ch in tail:
        if ch in _VALID:
            last = ch
    return last


def extract_direct(text: str) -> Optional[str]:
    """Protocol 1: first \\boxed{X}; fallback to first capital letter A-D."""
    if not text:
        return None
    m = _BOXED_PATTERN.search(text)
    if m:
        return m.group(1)
    return _first_capital_letter(text)


def extract_cot(text: str) -> Optional[str]:
    """Protocol 2: last \\boxed{X}; fallback to last capital letter A-D in tail."""
    if not text:
        return None
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1]
    return _last_capital_letter(text)


def vote_del_tom(answers: Sequence[Optional[str]]) -> Optional[str]:
    """Protocol 3: majority vote, alphabetic tiebreak; ignores None."""
    valid = [a for a in answers if a in _VALID]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    winners = sorted(c for c, n in counts.items() if n == max_count)
    return winners[0]
