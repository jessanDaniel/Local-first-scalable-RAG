import json
from types import SimpleNamespace

import pytest

from rag.eval import metrics
from rag.eval.runner import compare_runs, list_runs, load_run, new_run, save_run, score_questions, summarise
from rag.eval.sets import EvalQuestion, find_quote, load_set, resolve_gold, word_overlap
from rag.main import build_parser
from rag.models import Document, Section
from rag.store import db

# --- metrics ------------------------------------------------------------------


def test_metrics_on_a_known_ranking():
    ranked = ["a", "g1", "b", "g2", "c"]
    gold = {"g1", "g2", "g3"}
    assert metrics.first_rank(ranked, gold) == 2
    assert (metrics.hit_at(ranked, gold, 1), metrics.hit_at(ranked, gold, 2)) == (False, True)
    assert metrics.recall_at(ranked, gold, 5) == pytest.approx(2 / 3)
    assert metrics.reciprocal_rank(ranked, gold) == 0.5
    # DCG = 1/log2(3) + 1/log2(5); ideal = 1 + 1/log2(3) + 1/log2(4)
    assert metrics.ndcg_at(ranked, gold, 5) == pytest.approx((0.6309 + 0.4307) / (1 + 0.6309 + 0.5), abs=1e-3)


def test_metrics_when_nothing_is_found():
    assert metrics.first_rank(["a", "b"], {"g"}) is None
    assert metrics.reciprocal_rank(["a"], {"g"}) == 0.0
    assert metrics.ndcg_at(["a"], {"g"}, 10) == 0.0
    assert metrics.recall_at(["a"], set(), 10) == 0.0


# --- sets -------------------------------------------------------------------------


def write_set(tmp_path, rows) -> object:
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def row(id="q1", **overrides) -> dict:
    base = {"id": id, "collection": "c", "question": "What?", "type": "direct", "doc_id": "file:d", "quotes": ["x"]}
    return base | overrides


def test_load_set_validates_ids_and_gold(tmp_path):
    assert len(load_set(write_set(tmp_path, [row(), row("q2", doc_id=None, quotes=[])]))) == 2
    with pytest.raises(ValueError, match="duplicate id"):
        load_set(write_set(tmp_path, [row(), row()]))
    with pytest.raises(ValueError, match="needs both doc_id and quotes"):
        load_set(write_set(tmp_path, [row(quotes=[])]))
    with pytest.raises(ValueError, match="line 1"):
        load_set(write_set(tmp_path, [row(unknown_field=1)]))


def test_find_quote_statuses():
    chunks = [("c1", "Alpha beta gamma delta epsilon."), ("c2", "zeta eta theta iota kappa"), ("c3", "Alpha beta")]
    assert find_quote("beta  gamma\ndelta", chunks).status == "one chunk"  # whitespace-insensitive
    assert find_quote("Alpha beta", chunks).status == "several chunks"
    split = find_quote("gamma delta epsilon. zeta eta theta iota", chunks)
    assert (split.status, split.chunk_ids) == ("split", ["c1", "c2"])
    assert find_quote("not there at all", chunks).status == "missing"
    # split right after the first word, and right before the last one
    assert find_quote("epsilon. zeta eta theta iota", chunks).status == "split"
    assert find_quote("Alpha beta gamma delta epsilon. zeta", chunks).status == "split"


def test_word_overlap_ignores_stopwords():
    assert word_overlap("What is the refund window?", "The refund window is 30 days.") == 1.0
    assert word_overlap("What is the refund window?", "Shipping takes 5 days.") == 0.0


# --- scoring over a real (temporary) database ---------------------------------------


@pytest.fixture
def conn(tmp_path):
    from rag.chunk.chunker import Chunker
    from tests.test_dense import WordCounter

    connection = db.open_db(tmp_path / "rag.db")
    chunker = Chunker(WordCounter(), 60)
    for doc_id, text in {"file:a": "Refunds take 30 days.", "file:b": "Shipping takes 5 days."}.items():
        db.upsert_document(connection, Document(doc_id=doc_id, source="file", title=doc_id, collection_id="c"))
        sections = [Section.from_paragraphs(doc_id, 0, "1 S", 1, "1 S", [(text, None, None)])]
        db.replace_sections(connection, doc_id, sections)
        db.replace_chunks(connection, doc_id, chunker.chunk_document(sections))
    yield connection
    connection.close()


class FixedSearcher:
    """Returns the same ranking for every query: chunks of file:b first, then file:a."""

    def __init__(self, conn):
        self.chunks = db.get_chunks(conn, "file:b") + db.get_chunks(conn, "file:a")
        self.calls = []

    def search(self, query, k, collections=None):
        self.calls.append(collections)
        return [SimpleNamespace(chunk=c, score=1.0 - i / 10) for i, c in enumerate(self.chunks[:k])]


def test_score_and_summarise(conn):
    questions = [
        EvalQuestion("refund", "c", "How long do refunds take?", "direct", "file:a", ["Refunds take 30 days."]),
        EvalQuestion("none", "c", "Who is the CEO?", "missing_information"),
    ]
    gold, checks = resolve_gold(conn, questions)
    assert [c.status for c in checks] == ["one chunk"] and len(gold["refund"]) == 1
    searcher = FixedSearcher(conn)
    rows = score_questions(searcher, questions, gold, True, {"refund": "Refunds take 30 days."})
    assert searcher.calls == [["c"], ["c"]]  # the question's collection is the filter
    refund = rows[0]
    # "long" is not in the chunk: 2 of the 3 content words are
    assert (refund["first_rank"], refund["overlap"], refund["top1_doc"]) == (2, 0.67, "file:b")
    summary = summarise(rows)
    assert summary["overall"]["n"] == 1 and summary["overall"]["hit@1"] == 0 and summary["overall"]["hit@5"] == 1
    assert summary["overall"]["mrr"] == 0.5
    assert summary["unanswerable"] == {"n": 1, "mean_top1_score": 1.0}


def test_gold_can_span_documents_that_repeat_a_rule(conn):
    """also_in: the same rule stated in another document counts as gold too."""
    one = EvalQuestion("x", "c", "Delivery time?", "direct", "file:a", ["takes 5 days"])
    both = EvalQuestion("y", "c", "Delivery time?", "direct", "file:a", ["takes 5 days"], also_in=["file:b"])
    gold, checks = resolve_gold(conn, [one, both])
    assert [c.status for c in checks] == ["missing", "one chunk"]
    assert gold["x"] == [] and gold["y"] == [db.get_chunks(conn, "file:b")[0].chunk_id]


# --- saved runs ---------------------------------------------------------------------


def fake_run(set_name, ranks: dict[str, int | None], label=None, **config) -> dict:
    rows = [
        {
            "id": qid,
            "type": "t",
            "answerable": True,
            "gold": ["g"],
            "first_rank": rank,
            "ranked": (["x"] * (rank - 1) + ["g"]) if rank else ["x"],
            "top1_score": 0.5,
            "overlap": 0.5,
            "top1_doc": "d",
            "tags": [],
            "collection": "c",
            "scores": [],
        }
        for qid, rank in ranks.items()
    ]
    return new_run(set_name, {"index_type": "flat", **config}, rows, label)


def test_compare_reports_deltas_config_and_flips():
    a = fake_run("papers", {"q1": 1, "q2": 3, "q3": None}, query_instruction="x")
    b = fake_run("papers", {"q1": 1, "q2": 7, "q3": 2}, query_instruction="")
    text = "\n".join(compare_runs(a, b))
    assert "config query_instruction: 'x' -> ''" in text
    assert "q2" in text and "crosses top-5" in text
    assert "q3" in text and ">20 -> 2" in text
    assert "q1" not in text.split("questions whose")[1]
    with pytest.raises(ValueError, match="different sets"):
        compare_runs(a, fake_run("hr-demo", {"q1": 1}))


def test_latest_is_resolved_within_a_set(tmp_path):
    first = fake_run("papers", {"q1": 1}, label="one")
    first["created"] = "20260101T000000Z"
    other = fake_run("hr-demo", {"q1": 1})
    other["created"] = "20260102T000000Z"
    second = fake_run("papers", {"q1": 2}, label="two")
    second["created"] = "20260103T000000Z"
    for run in (first, other, second):
        run["run_id"] = f"{run['set']}-{run['created']}"
        save_run(run, tmp_path)
    assert load_run("papers:latest", tmp_path)["created"] == "20260103T000000Z"
    assert load_run("papers:latest~1", tmp_path)["created"] == "20260101T000000Z"
    assert len(list_runs(tmp_path, "papers")) == 2
    with pytest.raises(ValueError, match="ambiguous"):
        load_run("latest", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_run("papers:latest~5", tmp_path)


def test_eval_commands_parse():
    parser = build_parser()
    args = parser.parse_args(["eval", "run", "--set", "papers", "--label", "base", "--no-filter"])
    assert (args.handler, args.action, args.set, args.label, args.no_filter) == (
        "rag.eval.cli:run",
        "run",
        "papers",
        "base",
        True,
    )
    assert parser.parse_args(["eval", "compare", "papers:latest~1", "papers:latest"]).a == "papers:latest~1"
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "run"])  # --set is required
