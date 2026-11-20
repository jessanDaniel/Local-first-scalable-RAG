import pytest

from rag.fusion import rrf


def test_the_pitch_example():
    # A: dense 1, BM25 40.  B: dense 3, BM25 2.  C: dense 2, not in BM25.
    dense = ["A", "C", "B"]
    bm25 = ["x0", "B"] + [f"x{i}" for i in range(1, 38)] + ["A"]
    fused = {f.chunk_id: f for f in rrf({"dense": dense, "bm25": bm25})}
    assert fused["A"].ranks == {"dense": 1, "bm25": 40}
    assert fused["A"].score == pytest.approx(1 / 61 + 1 / 100)
    assert fused["B"].score == pytest.approx(1 / 63 + 1 / 62)
    assert fused["C"].score == pytest.approx(1 / 62) and fused["C"].ranks["bm25"] is None
    order = [f.chunk_id for f in rrf({"dense": dense, "bm25": bm25})]
    assert order.index("B") < order.index("A") < order.index("C")


def test_agreement_beats_a_single_first_place():
    fused = rrf({"dense": ["solo", "both"], "bm25": ["both"]})
    assert [f.chunk_id for f in fused] == ["both", "solo"]


def test_ties_are_deterministic():
    # Same score (rank 1 in one list each): the better best-rank ties, so the id decides.
    first = [f.chunk_id for f in rrf({"dense": ["b"], "bm25": ["a"]})]
    second = [f.chunk_id for f in rrf({"bm25": ["a"], "dense": ["b"]})]
    assert first == second == ["a", "b"]


def test_empty_lists():
    assert rrf({"dense": [], "bm25": []}) == []
    assert [f.chunk_id for f in rrf({"dense": ["a"], "bm25": []})] == ["a"]


def test_k_controls_how_much_first_place_counts():
    lists = {"dense": ["top", "mid"], "bm25": ["x", "mid"]}
    assert rrf(lists, k=60)[0].chunk_id == "mid"  # 2 x 1/62 > 1/61
    # k=0: top = 1/1, x = 1/1, mid = 1/2 + 1/2 = 1, a three-way tie. top and x have best rank 1
    # (mid only 2), and "top" < "x" by id.
    assert [f.chunk_id for f in rrf(lists, k=0)] == ["top", "x", "mid"]
