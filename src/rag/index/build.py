"""Build a dense index from the chunks in the database.

Each chunk is embedded as passage_text(title, section path, chunk text). Vectors
are cached by a hash of (model, exact input text), so rebuilding after a few
documents change only embeds the chunks whose input changed.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.chunk.tokens import TokenCounter
from rag.embed.embedder import passage_text
from rag.index.dense import DenseIndex, DenseManifest, set_current
from rag.store import db


class PassageEmbedder(Protocol):
    model_name: str
    dim: int
    device: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingCache:
    """Vectors keyed by sha1(model + input text), stored in one .npz file per model."""

    def __init__(self, path: Path):
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path.exists():
            data = np.load(path)
            self.vectors = dict(zip(data["keys"].tolist(), data["vectors"]))

    @staticmethod
    def key(model_name: str, text: str) -> str:
        return hashlib.sha1(f"{model_name}\n{text}".encode()).hexdigest()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.vectors), dtype=str)
        vectors = np.stack(list(self.vectors.values())) if self.vectors else np.zeros((0, 0), np.float32)
        temporary = self.path.with_suffix(".tmp.npz")
        np.savez(temporary, keys=keys, vectors=vectors)
        temporary.replace(self.path)


@dataclass
class BuildResult:
    build_id: str
    n_chunks: int
    n_embedded: int  # not found in the cache
    embed_seconds: float
    build_seconds: float
    index_bytes: int


def build_dense_index(
    conn,
    embedder: PassageEmbedder,
    counter: TokenCounter,
    indexes_dir: Path,
    cache: EmbeddingCache,
) -> BuildResult:
    start = time.perf_counter()
    rows = []
    for document in db.list_documents(conn):
        paths = {s.ord: s.path for s in db.get_sections(conn, document.doc_id)}
        for chunk in db.get_chunks(conn, document.doc_id):
            rows.append((chunk, passage_text(document.title, paths[chunk.section_ord], chunk.text, counter)))
    if not rows:
        raise ValueError("no chunks in the database: run `uv run python -m rag.chunk` first")
    versions = {chunk.chunker_version for chunk, _ in rows}
    if len(versions) != 1:
        raise ValueError(f"chunks come from several chunker versions {sorted(versions)}: rechunk first")

    keys = [EmbeddingCache.key(embedder.model_name, text) for _, text in rows]
    missing = [i for i, key in enumerate(keys) if key not in cache.vectors]
    embed_start = time.perf_counter()
    if missing:
        vectors = embedder.embed_passages([rows[i][1] for i in missing])
        for i, vector in zip(missing, vectors):
            cache.vectors[keys[i]] = vector
        cache.save()
    embed_seconds = time.perf_counter() - embed_start

    build_id = "dense-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    chunk_ids = [chunk.chunk_id for chunk, _ in rows]
    manifest = DenseManifest(build_id, embedder.model_name, versions.pop(), embedder.dim, len(rows))
    params = {"index_type": manifest.index_type, "dim": embedder.dim, "device": embedder.device,
              "passage_format": "title | section path\\nchunk text"}
    db.record_index_build(conn, build_id, "dense", embedder.model_name, manifest.chunker_version,
                          params, len(rows))
    try:
        index = DenseIndex.build(np.stack([cache.vectors[key] for key in keys]), chunk_ids, manifest)
        index.save(indexes_dir / build_id)
    except Exception:
        db.set_index_build_status(conn, build_id, "failed")
        raise
    db.set_index_build_status(conn, build_id, "ready")
    set_current(indexes_dir, build_id)
    return BuildResult(
        build_id=build_id,
        n_chunks=len(rows),
        n_embedded=len(missing),
        embed_seconds=embed_seconds,
        build_seconds=time.perf_counter() - start,
        index_bytes=(indexes_dir / build_id / "index.faiss").stat().st_size,
    )
