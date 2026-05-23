"""Generic answer extractors supporting variable option counts (A-Z)."""
import re
from collections import Counter
from typing import Optional, Sequence


def make_extractors(num_options: int):
    """Return (extract_direct, extract_cot, vote_del_tom) tuned to N options."""
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
