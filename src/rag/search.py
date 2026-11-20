"""Search the indexed chunks, dense or BM25, with an explain view.

    rag search "why divide attention scores by sqrt d_k" [-k 5] [--mode dense|bm25] [--explain]

Retrieval only: no prompt, no LLM. Each hit carries the chunk plus what a
citation needs (document title, section path, pages, source link).

    dense   bge-small embeddings + FAISS (the CURRENT build in data/indexes/dense/)
    bm25    hand-written BM25 (rag/index/bm25.py), built from SQLite in memory on each
            start; nothing on disk, so it can never be stale
"""

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.config import get_settings
from rag.models import Chunk
from rag.store import db

if TYPE_CHECKING:
    from rag.index.bm25 import BM25Index
    from rag.index.dense import DenseIndex


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk
    doc_title: str
    section_path: str
    source_uri: str | None


def make_hits(conn, ranked: list[tuple[str, float]]) -> list[Hit]:
    """(chunk_id, score) best first -> Hits with everything a citation needs, from SQLite."""
    chunks = {c.chunk_id: c for c in db.get_chunks_by_ids(conn, [chunk_id for chunk_id, _ in ranked])}
    hits = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        chunk = chunks[chunk_id]
        document = db.get_document(conn, chunk.doc_id)
        section = next(s for s in db.get_sections(conn, chunk.doc_id) if s.ord == chunk.section_ord)
        hits.append(Hit(rank, score, chunk, document.title, section.path, document.source_uri))
    return hits


class DenseSearcher:
    mode = "dense"

    def __init__(self, conn, index: "DenseIndex", embedder):
        self.conn = conn
        self.index = index
        self.embedder = embedder
        self.last_timings: dict[str, float] = {}

    def search(self, query: str, k: int = 5, collections: list[str] | None = None) -> list[Hit]:
        """Top-k hits; with `collections`, only chunks of documents in those collections."""
        start = time.perf_counter()
        allowed = db.chunk_ids_in_collections(self.conn, collections) if collections else None
        vector = self.embedder.embed_query(query)
        embedded = time.perf_counter()
        ranked = self.index.search(vector, k, allowed)
        searched = time.perf_counter()
        hits = make_hits(self.conn, ranked)
        self.last_timings = {
            "embed_query_ms": (embedded - start) * 1000,
            "search_ms": (searched - embedded) * 1000,
            "fetch_ms": (time.perf_counter() - searched) * 1000,
        }
        return hits


def bm25_documents(conn) -> list[tuple[str, str]]:
    """What BM25 indexes per chunk: "title | section path" + the chunk text, the same header the
    dense embedding gets, so a query naming the paper or the section can match it."""
    inputs = db.index_inputs(conn)
    texts = {c.chunk_id: c.text for c in db.get_chunks_by_ids(conn, [chunk_id for chunk_id, _, _ in inputs])}
    return [(chunk_id, f"{title} | {path}\n{texts[chunk_id]}") for chunk_id, title, path in inputs]


class BM25Searcher:
    mode = "bm25"

    def __init__(self, conn, index: "BM25Index"):
        self.conn = conn
        self.index = index
        self.last_timings: dict[str, float] = {}

    @classmethod
    def from_db(cls, conn) -> "BM25Searcher":
        from rag.index.bm25 import BM25Index

        start = time.perf_counter()
        searcher = cls(conn, BM25Index.build(bm25_documents(conn)))
        searcher.build_ms = (time.perf_counter() - start) * 1000
        return searcher

    def search(self, query: str, k: int = 5, collections: list[str] | None = None) -> list[Hit]:
        start = time.perf_counter()
        allowed = set(db.chunk_ids_in_collections(self.conn, collections)) if collections else None
        ranked = self.index.search(query, k, allowed)
        searched = time.perf_counter()
        hits = make_hits(self.conn, ranked)
        self.last_timings = {
            "search_ms": (searched - start) * 1000,
            "fetch_ms": (time.perf_counter() - searched) * 1000,
        }
        return hits


def print_hits(query: str, hits: list[Hit], searcher, explain: bool) -> None:
    if explain:
        print(f"Query:      {query}")
        if searcher.mode == "dense":
            manifest = searcher.index.manifest
            print(
                f"Embedded:   {searcher.embedder.query_instruction!r} + query  ({searcher.embedder.model_name}, "
                f"{searcher.embedder.device})"
            )
            print(
                f"Index:      {manifest.build_id} ({manifest.index_type}, {manifest.n_chunks} chunks, "
                f"chunker {manifest.chunker_version})"
            )
        else:
            from rag.index.tokenize import TOKENIZER_VERSION, query_terms

            index = searcher.index
            print(f"Terms:      {', '.join(query_terms(query)) or '(none)'}")
            print(
                f"Index:      BM25 in memory, {index.n_chunks} chunks, {len(index.postings)} terms, "
                f"avg length {index.avg_length:.0f} tokens, k1={index.k1}, b={index.b}, "
                f"tokenizer {TOKENIZER_VERSION}, built in {searcher.build_ms:.0f} ms"
            )
        print("Timings:    " + ", ".join(f"{name} {ms:.1f}" for name, ms in searcher.last_timings.items()))
        print()
    for hit in hits:
        c = hit.chunk
        pages = (
            f"p.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "") if c.page_start else "p.?"
        )
        print(f"{hit.rank:>2}. {hit.score:.4f}  {hit.doc_title[:60]}  ({c.doc_id}, {pages})")
        print(f"    {hit.section_path[:100]}")
        if explain:
            if searcher.mode == "bm25":
                parts = [p for p in searcher.index.explain(query, c.chunk_id) if p.tf]
                print("    " + "  ".join(f"{p.term}: tf {p.tf} × idf {p.idf:.2f} → {p.score:.2f}" for p in parts))
            text = " ".join(c.text.split())
            print(f"    [{c.chunk_type}, {c.n_tokens} tokens, {c.chunk_id}]")
            print(f"    {text[:400]}{' ...' if len(text) > 400 else ''}")
            print()


def make_searcher(conn, mode: str):
    """The searcher for a mode. Dense needs a CURRENT index (raises StaleIndexError) and loads the
    embedding model; BM25 builds its index from SQLite and loads nothing else."""
    if mode == "bm25":
        return BM25Searcher.from_db(conn)
    from rag.embed.embedder import Embedder
    from rag.index.dense import load_current

    settings = get_settings()
    index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model, conn)
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)
    return DenseSearcher(conn, index, embedder)


def run(parsed_args) -> int:
    from rag.index.current import StaleIndexError

    settings = get_settings()
    conn = db.open_db(settings.db_path)
    try:
        searcher = make_searcher(conn, parsed_args.mode)
    except StaleIndexError as exc:
        print(exc)
        return 1
    try:
        hits = searcher.search(parsed_args.query, parsed_args.k, parsed_args.collections)
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 1
    print_hits(parsed_args.query, hits, searcher, parsed_args.explain)
    return 0


def main() -> int:
    from rag.main import run_module

    return run_module(["search"])


if __name__ == "__main__":
    sys.exit(main())
