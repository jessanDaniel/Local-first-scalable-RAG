"""Dense search over the CURRENT index, with an explain view.

    rag search "why divide attention scores by sqrt d_k" [-k 5] [--explain]

Retrieval only: no prompt, no LLM. Each hit carries the chunk plus what a
citation needs (document title, section path, pages, source link).
"""

import sys
import time
from dataclasses import dataclass

from rag.config import get_settings
from rag.embed.embedder import Embedder
from rag.index.dense import DenseIndex, StaleIndexError, load_current
from rag.models import Chunk
from rag.store import db


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk
    doc_title: str
    section_path: str
    source_uri: str | None


class DenseSearcher:
    def __init__(self, conn, index: DenseIndex, embedder):
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
        chunks = {c.chunk_id: c for c in db.get_chunks_by_ids(self.conn, [chunk_id for chunk_id, _ in ranked])}
        hits = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = chunks[chunk_id]
            document = db.get_document(self.conn, chunk.doc_id)
            section = next(s for s in db.get_sections(self.conn, chunk.doc_id) if s.ord == chunk.section_ord)
            hits.append(Hit(rank, score, chunk, document.title, section.path, document.source_uri))
        self.last_timings = {
            "embed_query_ms": (embedded - start) * 1000,
            "search_ms": (searched - embedded) * 1000,
            "fetch_ms": (time.perf_counter() - searched) * 1000,
        }
        return hits


def print_hits(query: str, hits: list[Hit], searcher: DenseSearcher, explain: bool) -> None:
    if explain:
        manifest = searcher.index.manifest
        print(f"Query:      {query}")
        print(
            f"Embedded:   {searcher.embedder.query_instruction!r} + query  ({searcher.embedder.model_name}, "
            f"{searcher.embedder.device})"
        )
        print(
            f"Index:      {manifest.build_id} ({manifest.index_type}, {manifest.n_chunks} chunks, "
            f"chunker {manifest.chunker_version})"
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
            text = " ".join(c.text.split())
            print(f"    [{c.chunk_type}, {c.n_tokens} tokens, {c.chunk_id}]")
            print(f"    {text[:400]}{' ...' if len(text) > 400 else ''}")
            print()


def run(parsed_args) -> int:

    settings = get_settings()
    conn = db.open_db(settings.db_path)
    try:
        index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model, db.all_chunk_ids(conn))
    except StaleIndexError as exc:
        print(exc)
        return 1
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)
    searcher = DenseSearcher(conn, index, embedder)
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
