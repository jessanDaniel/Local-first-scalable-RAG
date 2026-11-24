import math

import pytest

from rag.index.bm25 import BM25Index

DOCS = [
    ("c0", "BLEU score of 34.81 on English to French"),
    ("c1", "BLEU score of 33.30 for the SMT baseline"),
    ("c2", "refunds are processed within seven business days"),
    ("c3", "BLEU BLEU BLEU BLEU"),
]


@pytest.fixture
def index():
    return BM25Index.build(DOCS)


def test_inverted_index_counts_terms_per_chunk(index):
    assert dict(index.postings["bleu"]) == {0: 1, 1: 1, 3: 4}
    assert index.postings["34.81"] == [(0, 1)]
    assert index.lengths[3] == 4


def test_idf_matches_the_formula_and_rare_terms_weigh_more(index):
    n, df = 4, 1
    assert index.idf("34.81") == pytest.approx(math.log(1 + (n - df + 0.5) / (df + 0.5)))
    assert index.idf("34.81") > index.idf("bleu") > 0


def test_score_matches_a_hand_computation(index):
    # The query "34.81" is three terms: "34.81", "34" and "81". Each occurs once in c0 and in no
    # other chunk, so each adds idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg)).
    length, avg = index.lengths[0], sum(index.lengths) / 4
    one_term = index.idf("34.81") * 1 * (1.2 + 1) / (1 + 1.2 * (1 - 0.75 + 0.75 * length / avg))
    expected = 3 * one_term
    [(chunk_id, score)] = index.search("34.81", k=5)
    assert chunk_id == "c0" and score == pytest.approx(expected)


def test_exact_number_beats_a_similar_chunk(index):
    ranked = [c for c, _ in index.search("What BLEU score was 34.81?", k=4)]
    assert ranked[0] == "c0"  # the rare token decides; c1 and c3 only share "bleu"/"score"


def test_term_frequency_saturates(index):
    # c3 says "bleu" 4 times but does not get 4x the score of a single mention.
    scores = dict(index.search("bleu", k=4))
    assert scores["c3"] < 4 * scores["c0"]


def test_unknown_terms_and_filters(index):
    assert index.search("transformer", k=5) == []
    assert [c for c, _ in index.search("bleu", k=5, allowed={"c1"})] == ["c1"]
    assert index.search("bleu", k=5, allowed=set()) == []
    with pytest.raises(ValueError):
        index.search("bleu", k=0)


def test_explain_sums_to_the_score(index):
    query = "BLEU score 34.81 transformer"
    [(top, score)] = index.search(query, k=1)
    parts = index.explain(query, top)
    assert sum(p.score for p in parts) == pytest.approx(score)
    assert {p.term: p.tf for p in parts} == {"bleu": 1, "score": 1, "34.81": 1, "34": 1, "81": 1, "transformer": 0}


# --- statistics over the searched chunks (step 24) ------------------------------------------------


def test_no_filter_and_a_filter_of_everything_score_the_same():
    from rag.index.bm25 import BM25Index

    docs = [("a", "leave policy for parents"), ("b", "annual leave and sick leave"), ("c", "password rules")]
    index = BM25Index.build(docs)
    assert index.search("sick leave", 3) == index.search("sick leave", 3, {"a", "b", "c"})


def test_another_collection_cannot_move_a_collections_ranking():
    """Adding documents full of "leave" elsewhere must not change scores inside {x, y}."""
    from rag.index.bm25 import BM25Index

    mine = [("x", "parental leave is twelve weeks"), ("y", "the leave request form and the policy text")]
    others = [(f"o{i}", "leave leave leave annual leave sick leave") for i in range(50)]
    alone = BM25Index.build(mine).search("parental leave", 2)
    together = BM25Index.build(mine + others).search("parental leave", 2, {"x", "y"})
    assert alone == together


def test_explain_uses_the_same_statistics_as_the_search():
    from rag.index.bm25 import BM25Index

    index = BM25Index.build([("x", "parental leave"), ("y", "leave form")] + [(f"o{i}", "leave") for i in range(20)])
    [(chunk, score)] = index.search("parental leave", 1, {"x", "y"})
    parts = index.explain("parental leave", chunk, {"x", "y"})
    assert abs(sum(p.score for p in parts) - score) < 1e-9 and {p.df for p in parts} == {1, 2}
