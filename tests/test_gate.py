import json

from rag.eval.gate import compare_flags, compare_retrieval, load_baseline, write_baseline


def snapshot(hit1=5, hit5=8, hit10=9, mrr=0.6, ranks=None):
    return {
        "set": {
            "n": 10,
            "metrics": {"hit@1": hit1, "hit@5": hit5, "hit@10": hit10, "mrr": mrr},
            "ranks": ranks if ranks is not None else {"a": 1, "b": 4, "c": 9},
        }
    }


def test_identical_numbers_pass():
    assert compare_retrieval(snapshot(), snapshot()) == ([], [])


def test_any_drop_fails_zero_tolerance():
    failures, _ = compare_retrieval(snapshot(), snapshot(hit5=7, mrr=0.599))
    assert "set hit@5: 8 -> 7" in failures and "set mrr: 0.6 -> 0.599" in failures


def test_a_question_leaving_the_top_5_fails_even_if_totals_hold():
    failures, improvements = compare_retrieval(snapshot(), snapshot(ranks={"a": 1, "b": 6, "c": 3}))
    assert failures == ["set b: rank 4 -> 6 (left the top 5)"]
    assert improvements == ["set c: rank 9 -> 3 (entered the top 5)"]


def test_moving_inside_the_top_5_is_not_a_failure():
    assert compare_retrieval(snapshot(), snapshot(ranks={"a": 2, "b": 1, "c": 9}))[0] == []


def test_improvements_pass_and_are_reported():
    failures, improvements = compare_retrieval(snapshot(), snapshot(hit1=6))
    assert failures == [] and improvements == ["set hit@1: 5 -> 6"]


def test_a_missing_set_fails():
    assert compare_retrieval(snapshot(), {})[0] == ["set: not measured"]


def test_flag_changes_fail_in_both_directions():
    before = {"run|q1|1": ["number_not_in_cited_source"], "run|q2|1": []}
    assert compare_flags(before, before) == []
    lost = compare_flags(before, {"run|q1|1": [], "run|q2|1": []})
    assert lost == ["flags changed on run q1 1: -number_not_in_cited_source"]
    new_false = compare_flags(before, {**before, "run|q2|1": ["entity_not_in_cited_document"]})
    assert new_false == ["flags changed on run q2 1: +entity_not_in_cited_document"]


def test_baseline_update_keeps_a_history_with_reasons(tmp_path):
    path = tmp_path / "baseline.json"
    first = write_baseline(snapshot(), {"k": []}, {"retriever": "hybrid"}, "initial", "abc123", None, path)
    second = write_baseline(snapshot(hit1=6), None, {"retriever": "hybrid"}, "better fusion", "def456", first, path)
    stored = load_baseline(path)
    assert stored == json.loads(path.read_text()) == second
    assert [h["reason"] for h in stored["history"]] == ["initial", "better fusion"]
    assert stored["flags"] == {"k": []}  # kept when the replay could not run
    assert stored["retrieval"]["set"]["metrics"]["hit@1"] == 6


def test_no_baseline_file():
    from pathlib import Path

    assert load_baseline(Path("/nonexistent/baseline.json")) is None


def test_label_doc_id_recovers_the_document_of_an_old_stored_source():
    from rag.eval.gate import label_doc_id

    label = "[4] Publication 463 Travel, Gift, and Car Expenses (file:tax-irs-p463-x) §1. Travel, pp.12-13"
    assert label_doc_id(label) == "file:tax-irs-p463-x"
    assert label_doc_id("[1] Attention Is All You Need (arxiv:1706.03762) §3 Model") == "arxiv:1706.03762"
    assert label_doc_id("no id here") == ""
