from scripts.eval.run_tombench import (
    build_direct_messages,
    build_cot_messages,
    build_user_prompt_zh,
    build_user_prompt_en,
)


def test_build_user_prompt_en():
    text = build_user_prompt_en(
        story="Alice put marble in box.",
        question="Where does Bob look?",
        opt_a="box", opt_b="basket", opt_c="bag", opt_d="cup",
    )
    assert "Story:" in text
    assert "Alice put marble in box." in text
    assert "Where does Bob look?" in text
    assert "A. box" in text
    assert "D. cup" in text


def test_build_user_prompt_zh():
    text = build_user_prompt_zh(
        story="小明把球放进盒子。",
        question="小红会去哪里找？",
        opt_a="盒子", opt_b="篮子", opt_c="书包", opt_d="杯子",
    )
    assert "故事：" in text
    assert "小明把球放进盒子。" in text
    assert "A. 盒子" in text


def test_build_direct_messages_has_system_prompt():
    msgs = build_direct_messages(
        story="s", question="q",
        opt_a="a", opt_b="b", opt_c="c", opt_d="d",
        language="en",
    )
    assert msgs[0]["role"] == "system"
    assert "\\boxed{X}" in msgs[0]["content"]
    assert "Do not include any explanation" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_build_cot_messages_allows_thinking():
    msgs = build_cot_messages(
        story="s", question="q",
        opt_a="a", opt_b="b", opt_c="c", opt_d="d",
        language="en",
    )
    assert "Think step by step" in msgs[0]["content"]
    assert "\\boxed{X}" in msgs[0]["content"]
