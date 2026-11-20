import hashlib
import re

import numpy as np
import pytest

from rag.chunk.chunker import Chunker
from rag.embed.embedder import (
    HEADER_TOKENS,
    Embedder,
    InputTooLongError,
    passage_text,
    trim_header,
)
from rag.index.build import EmbeddingCache, build_dense_index
from rag.index.dense import DenseIndex, DenseManifest, StaleIndexError, load_current
from rag.models import Document, Section
from rag.search import DenseSearcher
from rag.store import db


class WordCounter:
    max_tokens = 512

    def count(self, text: str) -> int:
        return len(text.split())


class FakeEmbedder:
    """Bag of words hashed into 64 dims, unit-length: chunks sharing words with a query score high."""

    model_name = "fake/bow-64"
    dim = 64
    device = "cpu"
    query_instruction = ""

    def __init__(self):
        self.embedded = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return vector / (np.linalg.norm(vector) or 1)

    def embed_passages(self, texts):
        self.embedded += len(texts)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, query):
        return self._vector(query)


# --- passage text -----------------------------------------------------------


def test_passage_text_puts_title_and_path_before_the_chunk():
    text = passage_text(
        "Attention Is All You Need", "5 Training > 5.4 Regularization", "We use dropout.", WordCounter()
    )
    assert text == "Attention Is All You Need | 5 Training > 5.4 Regularization\nWe use dropout."


def test_long_header_drops_outer_path_levels_first_then_title_words():
    title = " ".join(f"title{i}" for i in range(20))
    path = " > ".join(" ".join(f"level{n}word{i}" for i in range(15)) for n in range(4))
    header = trim_header(title, path, WordCounter())
    assert WordCounter().count(header) <= HEADER_TOKENS
    assert header.endswith(" ".join(f"level3word{i}" for i in range(15)))  # innermost section kept
    assert "level0word0" not in header
    assert header.startswith("title0 title1")


def test_embedder_refuses_inputs_the_model_would_truncate():
    embedder = Embedder.__new__(Embedder)  # skip loading a real model
    embedder.max_tokens = 5
    embedder.count = WordCounter().count
    embedder.query_instruction = ""
    with pytest.raises(InputTooLongError, match="would be dropped"):
        embedder.embed_passages(["one two three four five six"])


# --- FAISS index ------------------------------------------------------------


def random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def manifest(n: int, dim: int = 16) -> DenseManifest:
    return DenseManifest("dense-test", "fake/model", "v2-350", dim, n)


def test_faiss_matches_numpy_cosine():
    vectors = random_unit_vectors(50, 16)
    ids = [f"c{i}" for i in range(50)]
    index = DenseIndex.build(vectors, ids, manifest(50))
    query = random_unit_vectors(1, 16, seed=1)[0]

    hits = index.search(query, k=5)
    expected = np.argsort(-(vectors @ query))[:5]
    assert [chunk_id for chunk_id, _ in hits] == [ids[i] for i in expected]
    assert np.allclose([score for _, score in hits], (vectors @ query)[expected], atol=1e-6)


def test_k_larger_than_index_returns_everything():
    index = DenseIndex.build(random_unit_vectors(3, 16), ["a", "b", "c"], manifest(3))
    assert len(index.search(random_unit_vectors(1, 16)[0], k=10)) == 3


def test_save_load_round_trip_and_builds_are_never_overwritten(tmp_path):
    vectors = random_unit_vectors(10, 16)
    index = DenseIndex.build(vectors, [f"c{i}" for i in range(10)], manifest(10))
    index.save(tmp_path / "dense-test")
    loaded = DenseIndex.load(tmp_path / "dense-test")
    query = vectors[3]
    assert loaded.search(query, 3) == index.search(query, 3)
    assert loaded.manifest == index.manifest
    with pytest.raises(FileExistsError):
        index.save(tmp_path / "dense-test")


def test_wrong_vector_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        DenseIndex.build(random_unit_vectors(3, 8), ["a", "b", "c"], manifest(3, dim=16))


# --- build + search over a real (temporary) database --------------------------


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "rag.db")
    chunker = Chunker(WordCounter(), 60)
    papers = {
        "arxiv:1": (
            "Adam",
            [
                "Adam keeps moving averages of the gradient and its square.",
                "Bias correction fixes averages that start at zero.",
            ],
        ),
        "arxiv:2": (
            "ResNet",
            ["Residual connections let very deep networks train.", "Shortcut connections add the input to the output."],
        ),
    }
    for doc_id, (title, paragraphs) in papers.items():
        db.upsert_document(connection, Document(doc_id=doc_id, source="arxiv", title=title, parse_status="ok"))
        sections = [
            Section.from_paragraphs(doc_id, i, f"{i + 1} S", 1, f"{i + 1} S", [(p, i + 1, i + 1)])
            for i, p in enumerate(paragraphs)
        ]
        db.replace_sections(connection, doc_id, sections)
        db.replace_chunks(connection, doc_id, chunker.chunk_document(sections))
    yield connection
    connection.close()


def build(conn, tmp_path, embedder=None):
    embedder = embedder or FakeEmbedder()
    cache = EmbeddingCache(tmp_path / "embeddings" / "fake.npz")
    return embedder, build_dense_index(conn, embedder, WordCounter(), tmp_path / "indexes", cache)


def test_build_records_the_build_and_points_current_at_it(conn, tmp_path):
    _, result = build(conn, tmp_path)
    assert result.n_chunks == 4 and result.n_embedded == 4
    [record] = db.list_index_builds(conn, "dense")
    assert (record["build_id"], record["status"], record["n_chunks"]) == (result.build_id, "ready", 4)
    assert (tmp_path / "indexes" / "CURRENT").read_text() == result.build_id


def test_rebuild_reuses_cached_vectors(conn, tmp_path):
    build(conn, tmp_path)
    embedder, second = build(conn, tmp_path)
    assert second.n_embedded == 0 and embedder.embedded == 0


def test_search_returns_the_matching_chunk_with_citation_fields(conn, tmp_path):
    embedder, _ = build(conn, tmp_path)
    index = load_current(tmp_path / "indexes", embedder.model_name, conn)
    hits = DenseSearcher(conn, index, embedder).search("bias correction averages zero", k=2)
    top = hits[0]
    assert (top.rank, top.chunk.doc_id, top.doc_title, top.section_path) == (1, "arxiv:1", "Adam", "2 S")
    assert top.chunk.page_start == 2
    assert hits[0].score >= hits[1].score


def test_stale_index_is_refused(conn, tmp_path):
    embedder, _ = build(conn, tmp_path)
    with pytest.raises(StaleIndexError, match="settings now say"):
        load_current(tmp_path / "indexes", "other/model", conn)
    with conn:
        conn.execute("DELETE FROM documents WHERE doc_id = 'arxiv:2'")
    with pytest.raises(StaleIndexError, match="rebuild"):
        load_current(tmp_path / "indexes", embedder.model_name, conn)


def test_a_changed_title_makes_the_index_stale_even_with_the_same_chunks(conn, tmp_path):
    from dataclasses import replace

    embedder, _ = build(conn, tmp_path)
    load_current(tmp_path / "indexes", embedder.model_name, conn)  # current
    document = db.get_document(conn, "arxiv:1")
    db.upsert_document(conn, replace(document, title="Adam: A Method for Stochastic Optimization"))
    with pytest.raises(StaleIndexError, match="titles or section paths"):
        load_current(tmp_path / "indexes", embedder.model_name, conn)
    build(conn, tmp_path)  # rebuild: the changed header is embedded, the rest comes from the cache
    load_current(tmp_path / "indexes", embedder.model_name, conn)


def test_no_index_yet(tmp_path):
    with pytest.raises(StaleIndexError, match="no dense index"):
        load_current(tmp_path / "indexes", "fake/bow-64", db.open_db(tmp_path / "x.db"))


# --- collection filter --------------------------------------------------------


def test_filter_runs_inside_faiss_so_a_small_subset_still_gets_its_full_top_k():
    vectors = random_unit_vectors(100, 16)
    ids = [f"c{i}" for i in range(100)]
    index = DenseIndex.build(vectors, ids, manifest(100))
    query = vectors[0]
    allowed = ["c90", "c91", "c92", "c93", "c94"]  # far down the global ranking, most likely

    hits = index.search(query, 3, allowed)
    expected = sorted(allowed, key=lambda c: -float(vectors[int(c[1:])] @ query))[:3]
    assert [chunk_id for chunk_id, _ in hits] == expected
    assert len(index.search(query, 10, allowed)) == 5  # k larger than the subset returns the subset
    assert index.search(query, 3, []) == []
    assert index.search(query, 3, ["not-in-index"]) == []


def test_search_by_collection_never_returns_other_collections(tmp_path):
    connection = db.open_db(tmp_path / "rag.db")
    chunker = Chunker(WordCounter(), 60)
    docs = {
        ("arxiv:1", "papers"): "Adam keeps moving averages of the gradient. Bias correction fixes zero starts.",
        ("arxiv:2", "papers"): "Residual connections let very deep networks train.",
        ("file:handbook", "hr-demo"): "Employees get fifteen vacation days. Bias in hiring is reported to HR.",
    }
    for (doc_id, collection), text in docs.items():
        source = doc_id.split(":")[0]
        db.upsert_document(
            connection,
            Document(doc_id=doc_id, source=source, title=doc_id, parse_status="ok", collection_id=collection),
        )
        sections = [Section.from_paragraphs(doc_id, 0, "1 S", 1, "1 S", [(text, 1, 1)])]
        db.replace_sections(connection, doc_id, sections)
        db.replace_chunks(connection, doc_id, chunker.chunk_document(sections))
    embedder, _ = build(connection, tmp_path)
    index = load_current(tmp_path / "indexes", embedder.model_name, connection)
    searcher = DenseSearcher(connection, index, embedder)

    hr = searcher.search("bias correction", k=5, collections=["hr-demo"])
    assert [h.chunk.doc_id for h in hr] == ["file:handbook"]
    papers = searcher.search("bias correction", k=5, collections=["papers"])
    assert {h.chunk.doc_id for h in papers} == {"arxiv:1", "arxiv:2"}
    assert len(searcher.search("bias correction", k=5)) == 3  # no filter: everything
    with pytest.raises(KeyError, match="no collection"):
        searcher.search("bias", k=5, collections=["nope"])
    connection.close()


# --- hybrid (dense + BM25 + RRF) ------------------------------------------------


def test_hybrid_fuses_both_lists_and_records_where_each_hit_came_from(conn, tmp_path):
    from rag.search import BM25Searcher, HybridSearcher

    embedder, _ = build(conn, tmp_path)
    dense = DenseSearcher(conn, load_current(tmp_path / "indexes", embedder.model_name, conn), embedder)
    hybrid = HybridSearcher(conn, dense, BM25Searcher.from_db(conn), candidates=10, rrf_k=60)

    hits = hybrid.search("bias correction averages zero", k=4)
    assert hits[0].chunk.doc_id == "arxiv:1" and hits[0].section_path == "2 S"
    top = hybrid.last_ranks[hits[0].chunk.chunk_id]
    assert top == {"dense": 1, "bm25": 1}
    assert hits[0].score == pytest.approx(2 / 61)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(set(r) == {"dense", "bm25"} for r in hybrid.last_ranks.values())
    assert {"dense_embed_query_ms", "bm25_search_ms", "fuse_ms", "fetch_ms"} <= set(hybrid.last_timings)


def test_hybrid_keeps_a_chunk_only_bm25_finds(conn, tmp_path):
    from rag.search import BM25Searcher, HybridSearcher

    embedder, _ = build(conn, tmp_path)
    dense = DenseSearcher(conn, load_current(tmp_path / "indexes", embedder.model_name, conn), embedder)
    # dense returns only its top 1; BM25 returns everything that matches
    hybrid = HybridSearcher(conn, dense, BM25Searcher.from_db(conn), candidates=1, rrf_k=60)
    hits = hybrid.search("shortcut connections residual", k=4)
    origins = {h.chunk.chunk_id: hybrid.last_ranks[h.chunk.chunk_id] for h in hits}
    assert any(r["dense"] is None and r["bm25"] is not None for r in origins.values())
