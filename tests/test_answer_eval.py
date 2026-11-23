from types import SimpleNamespace

import pytest

from rag.eval.answers import (
    auto_outcome,
    check_answer,
    consistency,
    is_refusal,
    key_match,
    summarise_answers,
)
from rag.eval.labels import agreement_report, parse_sheet, pick_entries, render_entry
from rag.eval.sets import EvalQuestion


@pytest.mark.parametrize(
    ("text", "refused"),
    [
        ("The sources do not mention GPT-3's parameter count.", True),
        ("This is not covered in the provided context.", True),
        ("The provided sources contain no information about LoRA.", True),
        ("There is not enough information to answer.", True),
        ("The Acme Corp Employee Handbook does not specify the number of holidays.", True),  # missed in step 16
        ("The Transformer does not include recurrence [2].", False),  # a fact, not a refusal
        ("Adam does not provide bias correction for v_0 alone; it corrects both moments [1].", False),
        ("The model uses 8 heads [3].", False),
    ],
)
def test_refusal_detection_needs_the_sources_to_be_the_subject(text, refused):
    assert is_refusal(text) is refused


@pytest.mark.parametrize(
    ("answer", "expected", "match"),
    [
        ("The ensemble reached a BLEU score of 34.81 [1].", "34.81", True),
        ("It reached 34.8 BLEU [1].", "34.81", False),
        ("The error was 3.57% on the test set.", "3.57%", True),
        ("A total of 1,000 samples", "1000", True),
        ("It becomes softer [2].", "It becomes softer", True),
        ("The distribution gets sharper.", "It becomes softer", False),
        ("anything", None, None),
    ],
)
def test_key_match(answer, expected, match):
    assert key_match(answer, expected) is match


def test_auto_outcome_table():
    assert auto_outcome(True, True, None) == "refused_wrong"
    assert auto_outcome(True, False, True) == "correct?"
    assert auto_outcome(True, False, False) == "wrong?"
    assert auto_outcome(False, True, None) == "refused_ok"
    assert auto_outcome(False, False, None) == "answered?"


def fake_answer(text, cited, sources):
    return SimpleNamespace(
        text=text,
        sources=[SimpleNamespace(n=n, label=f"[{n}] Doc", text=f"text {n}", chunk_ids=ids) for n, ids in sources],
        citations=SimpleNamespace(cited=cited, invalid=[], n_sentences=2, n_cited_sentences=1),
        generation=SimpleNamespace(prompt_tokens=2900, completion_tokens=60, prompt_seconds=5.0, write_seconds=2.0),
        timings={"retrieve_s": 1.8, "generate_s": 7.0, "total_s": 8.9},
        warnings=[],
    )


def test_check_answer_records_gold_in_context_and_cited():
    q = EvalQuestion("q", "papers", "How many heads?", "identifier", "arxiv:1", ["h = 8"], "8")
    answer = fake_answer("It uses 8 heads [2].", [2], [(1, ["c1"]), (2, ["gold"])])
    result = check_answer(q, answer, ["gold"])
    assert (result["outcome"], result["gold_in_context"], result["gold_cited"]) == ("correct?", True, True)
    assert [s["cited"] for s in result["sources"]] == [False, True]
    assert (result["prompt_seconds"], result["write_seconds"]) == (5.0, 2.0)


def test_refusal_with_citations_is_flagged_for_unanswerable_questions():
    q = EvalQuestion("none", "papers", "GPT-3 size?", "unanswerable")
    result = check_answer(q, fake_answer("The sources do not mention GPT-3 [1].", [1], [(1, ["c1"])]), [])
    assert result["outcome"] == "refused_ok" and result["refused_with_citations"] is True


def attempt(run, outcome, cited_chunks, text="a"):
    return {
        "run": run,
        "outcome": outcome,
        "answer": text,
        "sources": [{"n": 1, "cited": True, "chunk_ids": cited_chunks, "gold": False, "label": "[1] D", "text": "t"}],
    }


def test_consistency_across_runs():
    same = consistency([attempt(1, "correct?", ["c1"]), attempt(2, "correct?", ["c1"])])
    assert same == {"same_outcome": True, "same_cited_chunks": True, "same_text": True}
    differ = consistency([attempt(1, "correct?", ["c1"]), attempt(2, "wrong?", ["c2"], text="b")])
    assert differ == {"same_outcome": False, "same_cited_chunks": False, "same_text": False}


def record(qid, answerable=True, outcomes=("correct?",), qtype="identifier"):
    attempts = []
    for i, outcome in enumerate(outcomes, 1):
        a = attempt(i, outcome, ["c1"])
        a.update(
            cited=[1],
            invalid_citations=[],
            sentences=2,
            cited_sentences=1,
            refused=outcome.startswith("refused"),
            refused_with_citations=False,
            key_match=outcome == "correct?" if answerable else None,
            gold_in_context=True if answerable else None,
            gold_cited=True if answerable else None,
            prompt_tokens=2900,
            answer_tokens=60,
            timings={"total_s": 9.0, "retrieve_s": 1.8},
            prompt_seconds=5.0,
            write_seconds=2.0,
        )
        attempts.append(a)
    return {
        "id": qid,
        "type": qtype,
        "collection": "papers",
        "question": f"{qid}?",
        "expected": "8",
        "answerable": answerable,
        "quotes": ["h = 8"] if answerable else [],
        "gold": ["c1"],
        "attempts": attempts,
        "consistency": consistency(attempts),
    }


def test_summary_counts():
    records = [
        record("a", outcomes=("correct?", "correct?")),
        record("b", outcomes=("wrong?", "correct?")),
        record("n", answerable=False, outcomes=("refused_ok", "refused_ok"), qtype="unanswerable"),
    ]
    s = summarise_answers(records)
    assert s["attempts"] == 6 and s["outcomes"] == {"correct?": 3, "refused_ok": 2, "wrong?": 1}
    assert s["unanswerable"] == {"refused": 2, "of": 2, "refused_with_citations": 0}
    assert s["consistency"]["same_outcome"] == 2  # a and n agree across runs, b does not
    assert s["seconds"]["prompt_median"] == 5.0


def test_sheet_round_trip_and_agreement():
    run = {
        "run_id": "papers-x",
        "set": "papers",
        "questions": [
            record("a"),
            record("b", outcomes=("wrong?",)),
            record("n", answerable=False, outcomes=("answered?",), qtype="unanswerable"),
        ],
    }
    entries = pick_entries([run], {"papers": 3})
    assert sorted(r["id"] for _, r, _ in entries) == ["a", "b", "n"]
    sheet = "".join(render_entry(run["run_id"], r, a, i) for i, (_, r, a) in enumerate(entries, 1))
    assert "Reference passage" in sheet and "not in the corpus" in sheet
    filled = {"a": "correct", "b": "correct", "n": "refused_ok"}
    for qid, verdict in filled.items():
        head = f"| {qid} | 1 -->"
        before, after = sheet.split(head, 1)
        after = after.replace("verdict: ", f"verdict: {verdict}", 1).replace("[1] supports: ", "[1] supports: no", 1)
        sheet = before + head + after
    labels, problems = parse_sheet(sheet)
    assert problems == [] and {lb["id"]: lb["verdict"] for lb in labels} == filled
    report = "\n".join(agreement_report(labels, {"papers-x": run}))
    assert "automatic outcome agrees with the label: 1/3" in report
    assert "b run 1: wrong? -> correct" in report and "n run 1: answered? -> refused_ok" in report
    assert "do not support the claim: 3" in report


def test_invalid_labels_are_reported():
    run = {"run_id": "r", "set": "papers", "questions": [record("a")]}
    sheet = render_entry("r", run["questions"][0], run["questions"][0]["attempts"][0], 1)
    labels, problems = parse_sheet(sheet.replace("verdict: ", "verdict: maybe"))
    assert problems and "maybe" in problems[0]


def test_cited_sources_are_shown_in_full_on_the_sheet():
    rec = record("a")
    rec["attempts"][0]["sources"][0]["text"] = "word " * 1000 + "THE SUPPORTING SENTENCE."
    assert "THE SUPPORTING SENTENCE." in render_entry("r", rec, rec["attempts"][0], 1)
