"""Unit tests for TomMcqRewardWorker reward computation.

Loads tom_mcq_reward_worker by file path to avoid running ROLL's
rewards/__init__.py which eagerly imports `ray` etc.
"""
import importlib.util
import math
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_WORKER_PATH = _HERE.parent / "roll" / "pipeline" / "rlvr" / "rewards" / "tom_mcq_reward_worker.py"

_spec = importlib.util.spec_from_file_location("tom_mcq_reward_worker", _WORKER_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_boxed_letter = _mod.extract_boxed_letter
sigmoid_window = _mod.sigmoid_window
tom_mcq_reward_fn = _mod.tom_mcq_reward_fn


def test_extract_boxed_basic():
    letter, fmt_ok = extract_boxed_letter("\\boxed{A}")
    assert letter == "A"
    assert fmt_ok is True


def test_extract_boxed_in_text():
    letter, fmt_ok = extract_boxed_letter("My answer is \\boxed{C}")
    assert letter == "C"
    assert fmt_ok is True


def test_extract_no_boxed():
    letter, fmt_ok = extract_boxed_letter("just text no boxed")
    assert fmt_ok is False


def test_extract_invalid_letter_inside_box():
    letter, fmt_ok = extract_boxed_letter("\\boxed{Z}")
    assert fmt_ok is False


def test_sigmoid_window_center_high():
    v = sigmoid_window(100, l_min=8, l_max=256, k=50)
    assert v > 0.9


def test_sigmoid_window_below_min_low():
    v = sigmoid_window(2, l_min=8, l_max=256, k=50)
    assert v < 0.25


def test_sigmoid_window_above_max_low():
    v = sigmoid_window(1000, l_min=8, l_max=256, k=50)
    assert v < 0.1


def test_reward_correct_short_format():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=5,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    assert r_len > 0.0
    assert r_total > 0.0


def test_reward_correct_but_overlong_low_reward():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=2000,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    assert r_len < 0.1
    assert r_total < 0.1


def test_reward_wrong_answer_zero():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{B}", response_token_count=5,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 0.0
    assert r_total == 0.0


def test_reward_no_boxed_zero():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="The answer is A.", response_token_count=5,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 0.0
    assert r_total == 0.0


def test_reward_boundaries():
    _, _, r_len_a, _ = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=8,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    _, _, r_len_b, _ = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=256,
        ground_truth="A", l_min=8, l_max=256, k=50,
    )
    assert 0.4 < r_len_a < 0.6
    assert 0.4 < r_len_b < 0.6


def test_reward_chinese_response():
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="答案是 \\boxed{C}", response_token_count=10,
        ground_truth="C", l_min=8, l_max=256, k=50,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    assert r_total > 0.0
