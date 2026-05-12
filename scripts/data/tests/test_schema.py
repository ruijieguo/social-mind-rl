from scripts.data.schema import TomRecord, ability_to_task


def test_tom_record_to_jsonl_dict():
    r = TomRecord(
        question_id="q1", source="tombench", language="en", task="False Belief",
        story="s", question="q",
        opt_a="a", opt_b="b", opt_c="c", opt_d="d", gold="A",
    )
    d = r.to_jsonl_dict()
    assert d["question_id"] == "q1"
    assert d["gold"] == "A"


def test_ability_to_task_known():
    assert ability_to_task("Belief: Location false beliefs") == "False Belief"
    assert ability_to_task("Non-literal Comm: Hinting") == "Non-literal Comm"


def test_ability_to_task_unknown_returns_other():
    assert ability_to_task("Some unknown ability") == "Other"
