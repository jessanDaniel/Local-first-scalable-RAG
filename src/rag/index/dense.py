"""Dense vector index: FAISS, saved as versioned builds on disk.

Exact search for now (IndexFlatIP): every query is scored against every chunk,
the same maths as the hand-written NumPy top-K of Phase 6, so it is the ground
truth that faster approximate indexes (HNSW, Stage 2) are measured against.
Vectors are unit-length, so inner product == cosine similarity.

The index holds only vectors and their row -> chunk_id map. Text, metadata and
pages always come from SQLite. Layout:

    data/indexes/dense/<build_id>/index.faiss
                                 /chunk_ids.json
                                 /manifest.json
    data/indexes/dense/CURRENT     <- build_id of the build searches use
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

# Re-exported: callers import these from here.
from rag.index.current import DenseManifest, StaleIndexError, check_current, current_build_dir  # noqa: F401


class DenseIndex:
    def __init__(self, index: faiss.Index, chunk_ids: list[str], manifest: DenseManifest):
        if index.ntotal != len(chunk_ids):
            raise ValueError(f"index has {index.ntotal} vectors but {len(chunk_ids)} chunk ids")
        self.index = index
        self.chunk_ids = chunk_ids
        self.manifest = manifest

    @classmethod
    def build(cls, vectors: np.ndarray, chunk_ids: list[str], manifest: DenseManifest) -> "DenseIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(chunk_ids), manifest.dim):
            raise ValueError(f"expected vectors of shape ({len(chunk_ids)}, {manifest.dim}), got {vectors.shape}")
        index = faiss.IndexFlatIP(manifest.dim)
        index.add(vectors)
        return cls(index, list(chunk_ids), manifest)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score), best first. k larger than the index returns everything."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, rows = self.index.search(query, min(k, self.index.ntotal))
        return [(self.chunk_ids[row], float(score)) for row, score in zip(rows[0], scores[0]) if row >= 0]

    def save(self, build_dir: Path) -> None:
        build_dir.mkdir(parents=True, exist_ok=False)  # a build is never overwritten
        faiss.write_index(self.index, str(build_dir / "index.faiss"))
        (build_dir / "chunk_ids.json").write_text(json.dumps(self.chunk_ids))
        (build_dir / "manifest.json").write_text(json.dumps(asdict(self.manifest), indent=2))

    @classmethod
    def load(cls, build_dir: Path) -> "DenseIndex":
        manifest = DenseManifest(**json.loads((build_dir / "manifest.json").read_text()))
        chunk_ids = json.loads((build_dir / "chunk_ids.json").read_text())
        return cls(faiss.read_index(str(build_dir / "index.faiss")), chunk_ids, manifest)


def set_current(indexes_dir: Path, build_id: str) -> None:
    """Point searches at a build. Written to a temp file then renamed, so readers never see half a name."""
    pointer = indexes_dir / "CURRENT"
    temporary = indexes_dir / "CURRENT.tmp"
    temporary.write_text(build_id)
    os.replace(temporary, pointer)


def load_current(indexes_dir: Path, embedding_model: str, db_chunk_ids: list[str]) -> DenseIndex:
    """The CURRENT build, after checking it still matches the configured model and the database."""
    check_current(indexes_dir, embedding_model, db_chunk_ids)
    return DenseIndex.load(current_build_dir(indexes_dir))
