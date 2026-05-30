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
    # Lowercase or non-letter inside boxed must fail format check.
    letter, fmt_ok = extract_boxed_letter("\\boxed{z}")
    assert fmt_ok is False
    letter, fmt_ok = extract_boxed_letter("\\boxed{1}")
    assert fmt_ok is False


def test_extract_extended_letters():
    # Stage 16+: support 6-opt EmoBench (A-F) and 15-opt Hi-ToM (A-O).
    for L in ["F", "K", "O", "Z"]:
        letter, fmt_ok = extract_boxed_letter(f"reasoning... \\boxed{{{L}}}")
        assert fmt_ok is True
        assert letter == L


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


def test_reward_long_cot_with_widened_l_max():
    # Hi-ToM order_3 needs ~400-token CoT. With l_max=256 r_len collapses;
    # with l_max=512 it stays ~1.0 (the Stage 16 task-aware widening).
    _, _, r_len_short, _ = tom_mcq_reward_fn(
        response="reasoning ... \\boxed{K}", response_token_count=400,
        ground_truth="K", l_min=8, l_max=256, k=50,
    )
    _, _, r_len_long, _ = tom_mcq_reward_fn(
        response="reasoning ... \\boxed{K}", response_token_count=400,
        ground_truth="K", l_min=8, l_max=512, k=50,
    )
    assert r_len_short < 1e-6
    assert r_len_long > 0.99


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


def test_weighted_sum_correct_answer():
    """Stage 9+: weighted_sum should give r_out_weight when format+answer correct."""
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="\\boxed{A}", response_token_count=200,
        ground_truth="A", l_min=100, l_max=600, k=50,
        aggregation="weighted_sum",
        r_fmt_weight=0.05, r_out_weight=0.85, r_len_weight=0.10,
    )
    assert r_fmt == 1.0
    assert r_out == 1.0
    # 0.05 + 0.85 + 0.10 * r_len ≈ 0.95 + 0.10 * 0.95 ≈ 1.0
    assert 0.95 < r_total <= 1.0


def test_weighted_sum_wrong_format_keeps_some_credit():
    """Wrong format gives 0 in multiplicative; weighted_sum keeps r_len credit."""
    r_fmt_m, _, _, r_total_m = tom_mcq_reward_fn(
        response="The answer is A.", response_token_count=200,
        ground_truth="A", l_min=100, l_max=600, k=50,
        aggregation="multiplicative",
    )
    r_fmt_w, _, r_len_w, r_total_w = tom_mcq_reward_fn(
        response="The answer is A.", response_token_count=200,
        ground_truth="A", l_min=100, l_max=600, k=50,
        aggregation="weighted_sum",
        r_fmt_weight=0.05, r_out_weight=0.85, r_len_weight=0.10,
    )
    assert r_fmt_m == 0.0
    assert r_total_m == 0.0  # multiplicative: zero out
    assert r_fmt_w == 0.0
    # weighted_sum: still gets 0.10 * r_len credit
    assert 0.05 < r_total_w < 0.15  # ~0.10 * r_len_high


def test_weighted_sum_correct_but_format_off_still_partial():
    """If model answers correctly with bad format, weighted_sum gives partial credit."""
    # Note: r_out requires fmt_ok per current logic, so this stays 0.
    # But weighted_sum still gives r_len component (vs multiplicative giving 0).
    r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
        response="A is correct", response_token_count=150,
        ground_truth="A", l_min=100, l_max=600, k=50,
        aggregation="weighted_sum",
        r_fmt_weight=0.05, r_out_weight=0.85, r_len_weight=0.10,
    )
    assert r_fmt == 0.0
    assert r_out == 0.0
    # Only r_len contributes
    assert 0.05 < r_total < 0.15



# --- apply_reward_override (JSON override; ROLL drops YAML reward keys) -------
apply_reward_override = _mod.apply_reward_override


def test_override_none_returns_base_unchanged():
    base = {"l_min": 8.0, "l_max": 256.0, "aggregation": "multiplicative"}
    assert apply_reward_override(base, None) is base
    assert apply_reward_override(base, {}) is base


def test_override_merges_and_coerces_types():
    base = {
        "l_min": 8.0, "l_max": 256.0, "k": 50.0, "l_max_long": 256.0,
        "l_max_short": 256.0, "aggregation": "multiplicative",
        "r_fmt_weight": 0.05, "r_out_weight": 0.85, "r_len_weight": 0.10,
    }
    out = apply_reward_override(base, {
        "l_max": 2048, "l_max_long": 4096, "l_max_short": 512,
        "aggregation": "weighted_sum", "r_out_weight": 0.90, "r_len_weight": 0.05,
    })
    assert out["l_max"] == 2048.0 and isinstance(out["l_max"], float)
    assert out["aggregation"] == "weighted_sum"
    assert out["r_out_weight"] == 0.90 and out["r_len_weight"] == 0.05
    assert out["l_min"] == 8.0  # untouched key preserved
    assert base["l_max"] == 256.0  # input not mutated
