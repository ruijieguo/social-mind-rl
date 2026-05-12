import json
from scripts.data.synth_tomtype import parse_synth_response


def test_parse_well_formed_json_object():
    raw = json.dumps({
        "story": "Alice sees a marble go into the basket.",
        "question": "Where will Bob look for the marble?",
        "options": {"A": "basket", "B": "box", "C": "cupboard", "D": "fridge"},
        "answer": "A",
    })
    rec = parse_synth_response(raw)
    assert rec is not None
    assert rec.story.startswith("Alice")
    assert rec.opt_a == "basket"
    assert rec.gold == "A"


def test_parse_extracts_json_from_markdown_fence():
    raw = "Here is the question:\n```json\n" + json.dumps({
        "story": "s", "question": "q",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer": "B",
    }) + "\n```"
    rec = parse_synth_response(raw)
    assert rec is not None
    assert rec.gold == "B"


def test_parse_returns_none_on_missing_field():
    raw = json.dumps({"story": "s", "question": "q", "options": {"A": "a"}})
    assert parse_synth_response(raw) is None


def test_parse_returns_none_on_garbage():
    assert parse_synth_response("not json at all") is None
