"""Build or list dense indexes.

rag index build     # embed all chunks (cached), write a new build, make it CURRENT
rag index list      # builds recorded in the database
"""

import sys

from rag.chunk.tokens import HFTokenCounter
from rag.config import get_settings
from rag.embed.embedder import Embedder
from rag.index.build import EmbeddingCache, build_dense_index
from rag.store import db


def run(args) -> int:
    command = args.action

    settings = get_settings()
    conn = db.open_db(settings.db_path)
    if command == "list":
        current_file = settings.data_dir / "indexes" / "dense" / "CURRENT"
        current = current_file.read_text().strip() if current_file.exists() else None
        for build in db.list_index_builds(conn, "dense"):
            mark = "*" if build["build_id"] == current else " "
            print(
                f"{mark} {build['build_id']}  {build['status']:8} {build['n_chunks']:6} chunks  "
                f"{build['embedding_model']}  chunker {build['chunker_version']}  {build['params'].get('device')}"
            )
        return 0

    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)
    counter = HFTokenCounter(settings.embedding_model)
    cache = EmbeddingCache(settings.data_dir / "embeddings" / f"{settings.embedding_model.replace('/', '__')}.npz")
    result = build_dense_index(conn, embedder, counter, settings.data_dir / "indexes" / "dense", cache)
    rate = result.n_embedded / result.embed_seconds if result.n_embedded else 0
    print(f"build {result.build_id} is CURRENT")
    print(
        f"  chunks: {result.n_chunks}  newly embedded: {result.n_embedded} "
        f"({result.n_chunks - result.n_embedded} from cache)"
    )
    print(
        f"  embedding: {result.embed_seconds:.2f}s on {embedder.device}" + (f" ({rate:.0f} chunks/s)" if rate else "")
    )
    print(f"  total build: {result.build_seconds:.2f}s   index file: {result.index_bytes / 1024:.0f} KB")
    return 0


def main() -> int:
    from rag.main import run_module

    return run_module(["index"])


if __name__ == "__main__":
    sys.exit(main())
