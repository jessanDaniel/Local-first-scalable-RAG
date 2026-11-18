"""Chunk every parsed document into the database, then report what came out.

    uv run python -m rag.chunk                       # all ok/suspect documents
    uv run python -m rag.chunk --show arxiv:1706.03762 [--limit 20]

Chunks are replaced per document, so rerunning is safe. --show is the debug
view: each chunk's id, type, section path, pages, token count and text.
"""

import argparse
import statistics
import sys
import time
from collections import Counter

from rag.chunk.chunker import Chunker
from rag.chunk.tokens import HFTokenCounter
from rag.config import get_settings
from rag.store import db

TOKEN_BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 351)]


def show(conn, doc_id: str, limit: int | None) -> int:
    document = db.get_document(conn, doc_id)
    if document is None:
        print(f"no document {doc_id}")
        return 1
    paths = {s.ord: s.path for s in db.get_sections(conn, doc_id)}
    chunks = db.get_chunks(conn, doc_id)
    print(f"{doc_id}: {document.title}\n{len(chunks)} chunks\n")
    for chunk in chunks[:limit]:
        pages = f"p.{chunk.page_start}-{chunk.page_end}" if chunk.page_start else "p.?"
        print(f"[{chunk.ord}] {chunk.chunk_id} {chunk.chunk_type} {pages} {chunk.n_tokens} tokens")
        print(f"    {paths[chunk.section_ord]}")
        text = " ".join(chunk.text.split())
        print(f"    {text[:300]}{' ...' if len(text) > 300 else ''}\n")
    return 0


def main() -> int:
    args = argparse.ArgumentParser(description="Chunk parsed documents.")
    args.add_argument("--show", metavar="DOC_ID", help="print one document's chunks instead")
    args.add_argument("--limit", type=int, help="with --show: only the first N chunks")
    parsed_args = args.parse_args()

    settings = get_settings()
    conn = db.open_db(settings.db_path)
    if parsed_args.show:
        return show(conn, parsed_args.show, parsed_args.limit)

    chunker = Chunker(HFTokenCounter(settings.embedding_model), settings.chunk_max_tokens)
    documents = [d for d in db.list_documents(conn) if d.parse_status in ("ok", "suspect")]
    all_tokens: list[int] = []
    types: Counter = Counter()
    start = time.perf_counter()
    for document in documents:
        chunks = chunker.chunk_document(db.get_sections(conn, document.doc_id))
        db.replace_chunks(conn, document.doc_id, chunks)
        tokens = [c.n_tokens for c in chunks]
        all_tokens.extend(tokens)
        types.update(c.chunk_type for c in chunks)
        print(
            f"{document.doc_id:24} {len(chunks):4} chunks  tokens median {statistics.median(tokens):5.0f} "
            f"max {max(tokens):4}  {document.title[:45]}"
        )
    elapsed = time.perf_counter() - start

    print(f"\n{len(documents)} documents, {len(all_tokens)} chunks ({chunker.version}, "
          f"{settings.embedding_model} tokens) in {elapsed:.1f}s")
    print("types: " + ", ".join(f"{t} {n}" for t, n in types.most_common()))
    print(f"tokens: median {statistics.median(all_tokens):.0f}, mean {statistics.mean(all_tokens):.0f}, "
          f"max {max(all_tokens)} (limit {settings.chunk_max_tokens})")
    for low, high in TOKEN_BUCKETS:
        n = sum(low <= t < high for t in all_tokens)
        print(f"  {low:>3}-{high - 1:<3} tokens: {n:5} ({n / len(all_tokens):.0%})")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
