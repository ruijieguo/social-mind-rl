import pytest
from scripts.eval.extractors import extract_direct, extract_cot, vote_del_tom


# ---- Direct protocol: \boxed{X} first match ----

def test_direct_simple_boxed():
    assert extract_direct(r"\boxed{A}") == "A"


def test_direct_with_whitespace():
    assert extract_direct(r"  \boxed{B}  ") == "B"


def test_direct_picks_first_when_multiple():
    assert extract_direct(r"\boxed{A} then \boxed{C}") == "A"


def test_direct_fallback_first_capital_letter():
    # No boxed, fall back to first standalone capital letter A-D
    assert extract_direct("The answer is C.") == "C"


def test_direct_returns_none_when_nothing():
    assert extract_direct("blah blah") is None


def test_direct_invalid_letter_in_box_fallsback():
    # \boxed{Z} is invalid; should fall back to letter search
    assert extract_direct(r"\boxed{Z} but actually A") == "A"


# ---- CoT protocol: \boxed{X} last match ----

def test_cot_picks_last_boxed():
    text = "I think A first.\nWait, actually \\boxed{D}."
    assert extract_cot(text) == "D"


def test_cot_picks_last_when_multiple():
    assert extract_cot(r"\boxed{A} ... \boxed{B} ... \boxed{C}") == "C"


def test_cot_fallback_last_capital_letter_in_tail():
    text = "Long reasoning... final answer: B"
    assert extract_cot(text) == "B"


def test_cot_returns_none_when_nothing():
    assert extract_cot("just blabber") is None


# ---- DEL-ToM voting ----

def test_del_tom_majority_vote():
    answers = ["A", "A", "B", "A", "C", "A", "B", "A"]
    assert vote_del_tom(answers) == "A"


def test_del_tom_tie_breaks_alphabetically():
    answers = ["A", "B", "A", "B"]
    # Tie between A and B; alphabetic
    assert vote_del_tom(answers) == "A"


def test_del_tom_ignores_none():
    answers = ["A", None, "B", "A", None]
    assert vote_del_tom(answers) == "A"


def test_del_tom_all_none_returns_none():
    assert vote_del_tom([None, None, None]) is None
