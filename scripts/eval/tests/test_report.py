from scripts.eval.report import aggregate_results, format_markdown_table


def _sample_record(qid, lang, task, gold, pred, model, protocol):
    return {
        "question_id": qid,
        "language": lang,
        "task": task,
        "gold": gold,
        "pred": pred,
        "model": model,
        "protocol": protocol,
        "correct": pred == gold,
    }


def test_aggregate_basic_overall():
    rs = [
        _sample_record("1", "en", "False Belief", "A", "A", "m", "direct"),
        _sample_record("2", "en", "False Belief", "B", "C", "m", "direct"),
        _sample_record("3", "zh", "False Belief", "A", "A", "m", "direct"),
    ]
    agg = aggregate_results(rs)
    cell = agg[("m", "direct")]
    assert abs(cell["overall"] - 2/3) < 1e-6
    assert abs(cell["en"] - 1/2) < 1e-6
    assert abs(cell["zh"] - 1.0) < 1e-6


def test_aggregate_per_task_split():
    rs = [
        _sample_record("1", "en", "False Belief", "A", "A", "m", "direct"),
        _sample_record("2", "en", "Faux-pas", "B", "C", "m", "direct"),
        _sample_record("3", "en", "Faux-pas", "A", "A", "m", "direct"),
    ]
    agg = aggregate_results(rs)
    cell = agg[("m", "direct")]
    assert cell["task"]["False Belief"] == 1.0
    assert cell["task"]["Faux-pas"] == 0.5


def test_aggregate_multiple_models_protocols():
    rs = [
        _sample_record("1", "en", "False Belief", "A", "A", "m1", "direct"),
        _sample_record("1", "en", "False Belief", "A", "B", "m1", "cot"),
        _sample_record("1", "en", "False Belief", "A", "A", "m2", "direct"),
    ]
    agg = aggregate_results(rs)
    assert ("m1", "direct") in agg
    assert ("m1", "cot") in agg
    assert ("m2", "direct") in agg
    assert agg[("m1", "direct")]["overall"] == 1.0
    assert agg[("m1", "cot")]["overall"] == 0.0


def test_format_markdown_table_contains_headers():
    agg = {
        ("qwen3-8b-nt", "direct"): {
            "overall": 0.5349, "en": 0.55, "zh": 0.52,
            "task": {"False Belief": 0.6, "Faux-pas": 0.4},
        },
    }
    md = format_markdown_table(agg)
    assert "qwen3-8b-nt" in md
    assert "direct" in md
    assert "0.5349" in md
