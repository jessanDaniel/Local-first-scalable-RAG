"""Which dense build is CURRENT, and is it still valid? Reads only the JSON files, no FAISS.

Kept apart from dense.py so `rag status` can check the index in milliseconds
without importing FAISS or NumPy. dense.load_current uses the same check.

An index is valid only if it was embedded from what the database holds now:
the same model, the same chunk ids (chunk text is in the id), and the same
document titles and section paths, which go into every embedding as the
"title | section path" header but are not part of the chunk id.
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rag.store import db

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
    # sha1 over every chunk's (chunk_id, document title, section path). None in builds made
    # before this check existed.
    inputs_fingerprint: str | None = None


def inputs_fingerprint(rows: Iterable[tuple[str, str, str]]) -> str:
    """Fingerprint of (chunk_id, title, section path) rows, independent of their order."""
    digest = hashlib.sha1()
    for row in sorted(rows):
        digest.update("\t".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def database_state(conn: sqlite3.Connection) -> tuple[list[str], str]:
    """What an index must match: the chunk ids, and the fingerprint of their embedding headers."""
    inputs = db.index_inputs(conn)
    return [chunk_id for chunk_id, _, _ in inputs], inputs_fingerprint(inputs)


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
    manifest: DenseManifest,
    chunk_ids: list[str],
    embedding_model: str,
    db_chunk_ids: list[str],
    db_fingerprint: str,
) -> str | None:
    """Why this build no longer matches the settings and the database, or None if it does."""
    if manifest.embedding_model != embedding_model:
        return (
            f"index {manifest.build_id} was built with {manifest.embedding_model}, "
            f"settings now say {embedding_model}: {REBUILD}"
        )
    if set(chunk_ids) != set(db_chunk_ids):
        return (
            f"index {manifest.build_id} holds {len(chunk_ids)} chunks, the database has "
            f"{len(db_chunk_ids)} different or changed ones: {REBUILD}"
        )
    if manifest.inputs_fingerprint is None:
        return f"index {manifest.build_id} predates the title/section check: {REBUILD}"
    if manifest.inputs_fingerprint != db_fingerprint:
        return (
            f"index {manifest.build_id} was embedded with document titles or section paths that have "
            f"since changed: {REBUILD}"
        )
    return None


def check_current(indexes_dir: Path, embedding_model: str, conn: sqlite3.Connection) -> DenseManifest:
    """The CURRENT build's manifest; raises StaleIndexError if there is none or it is out of date."""
    build_dir = current_build_dir(indexes_dir)
    if build_dir is None:
        raise StaleIndexError("no dense index built yet: run `rag index build`")
    manifest = read_manifest(build_dir)
    db_chunk_ids, fingerprint = database_state(conn)
    reason = stale_reason(manifest, read_chunk_ids(build_dir), embedding_model, db_chunk_ids, fingerprint)
    if reason:
        raise StaleIndexError(reason)
    return manifest
