"""Which dense build is CURRENT, and is it still valid? Reads only the JSON files, no FAISS.

Kept apart from dense.py so `rag status` can check the index in milliseconds
without importing FAISS or NumPy. dense.load_current uses the same check.
"""

import json
from dataclasses import dataclass
from pathlib import Path

REBUILD = "rebuild with `rag index build`"


class StaleIndexError(RuntimeError):
    """The index was built from different chunks or a different model than the database now has."""


@dataclass
class DenseManifest:
    build_id: str
    embedding_model: str
    chunker_version: str
    dim: int
    n_chunks: int
    metric: str = "inner_product"
    index_type: str = "IndexFlatIP"


def current_build_dir(indexes_dir: Path) -> Path | None:
    pointer = indexes_dir / "CURRENT"
    if not pointer.exists():
        return None
    return indexes_dir / pointer.read_text().strip()


def read_manifest(build_dir: Path) -> DenseManifest:
    return DenseManifest(**json.loads((build_dir / "manifest.json").read_text()))


def read_chunk_ids(build_dir: Path) -> list[str]:
    return json.loads((build_dir / "chunk_ids.json").read_text())


def stale_reason(
    manifest: DenseManifest, chunk_ids: list[str], embedding_model: str, db_chunk_ids: list[str]
) -> str | None:
    """Why this build no longer matches the settings and the database, or None if it does."""
    if manifest.embedding_model != embedding_model:
        return (f"index {manifest.build_id} was built with {manifest.embedding_model}, "
                f"settings now say {embedding_model}: {REBUILD}")
    if set(chunk_ids) != set(db_chunk_ids):
        return (f"index {manifest.build_id} holds {len(chunk_ids)} chunks, the database has "
                f"{len(db_chunk_ids)} different or changed ones: {REBUILD}")
    return None


def check_current(indexes_dir: Path, embedding_model: str, db_chunk_ids: list[str]) -> DenseManifest:
    """The CURRENT build's manifest; raises StaleIndexError if there is none or it is out of date."""
    build_dir = current_build_dir(indexes_dir)
    if build_dir is None:
        raise StaleIndexError("no dense index built yet: run `rag index build`")
    manifest = read_manifest(build_dir)
    reason = stale_reason(manifest, read_chunk_ids(build_dir), embedding_model, db_chunk_ids)
    if reason:
        raise StaleIndexError(reason)
    return manifest
