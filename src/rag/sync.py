"""Keep a collection in step with its folders and uploads: only what changed is redone.

    rag sync COLLECTION                       # rescan its folders and uploads
    rag sync COLLECTION --add-folder PATH     # register a folder, then sync
    rag sync COLLECTION --remove-folder PATH  # forget a folder and its documents, then sync

What a sync does, in order (each step reported as an event, for the CLI and the API's progress):

 1. Sources: the collection's registered folders (table source_folders) and its upload folder
    data/raw/uploads/<collection>/. A folder that doesn't exist right now (an unplugged disk) is
    reported and left alone: its documents are kept, never deleted.
 2. Files: every folder walked recursively (rag.ingest.find_documents: hidden, temporary, empty,
    unsupported, too large and duplicate files are skipped, each with its reason).
 3. Ingest: a file whose content hash is unchanged is skipped; a new or changed file is parsed and
    stored. One file failing doesn't stop the others.
 4. Removals: a document whose original file lived in one of these (existing) folders and is gone
    now is deleted with its sections and chunks, and its stored copy removed. Documents added any
    other way (rag ingest files, arXiv) are never touched by a sync.
 5. Chunk: only the documents parsed in step 3.
 6. Index: if anything changed, a new dense build from the embedding cache (only new chunks are
    embedded), made CURRENT; builds beyond the newest 3 are deleted (they can always be rebuilt).

The running server swaps the new indexes in afterwards (Pipeline.reload); BM25 is rebuilt from
SQLite there in milliseconds.
"""

import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import Settings, get_settings
from rag.store import db

KEEP_BUILDS = 3
UPLOAD_PREFIX = "upload"


class SyncCancelled(Exception):
    """The sync was asked to stop; what was finished is kept, and the next sync goes on from there."""


@dataclass
class SyncEvent:
    kind: str  # source | missing-folder | skipped | unchanged | added | updated | failed | removed | chunked | indexed
    name: str
    detail: str = ""


@dataclass
class SyncReport:
    collection: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    missing_folders: list[str] = field(default_factory=list)
    n_chunks: int = 0  # chunks written for re-parsed documents
    build_id: str | None = None
    n_embedded: int = 0
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def uploads_dir(settings: Settings, collection: str) -> Path:
    return settings.data_dir / "raw" / "uploads" / collection


def under(path: str | None, folder: Path) -> bool:
    if not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


def prune_builds(indexes_dir: Path, keep: int = KEEP_BUILDS) -> list[str]:
    """Delete dense builds beyond the newest `keep` (never the CURRENT one). Rebuildable data only."""
    current = (indexes_dir / "CURRENT").read_text().strip() if (indexes_dir / "CURRENT").exists() else None
    builds = sorted((p for p in indexes_dir.glob("dense-*") if p.is_dir()), key=lambda p: p.name, reverse=True)
    removed = []
    for build in builds[keep:]:
        if build.name != current:
            shutil.rmtree(build)
            removed.append(build.name)
    return removed


def sync_collection(
    settings: Settings,
    collection: str,
    on_event: Callable[[SyncEvent], None] | None = None,
    cancel: threading.Event | None = None,
    embedder=None,
    counter=None,
    rebuild: bool = False,
) -> SyncReport:
    """Bring `collection` in step with its folders and uploads. embedder / counter: pass loaded ones
    to reuse them (the server); otherwise they are loaded when first needed. rebuild: make a new
    index even if this sync changed nothing (documents were removed before it, say)."""
    from rag.chunk.chunker import Chunker
    from rag.ingest import FoundFile, Ingestor, find_documents
    from rag.parse.base import ParseError

    report = SyncReport(collection)
    started = time.perf_counter()

    def emit(kind: str, name: str, detail: str = "") -> None:
        if on_event is not None:
            on_event(SyncEvent(kind, name, detail))

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise SyncCancelled(f"sync of {collection} cancelled")

    ingestor = Ingestor(settings, collection=collection)
    conn = ingestor.conn
    try:
        if collection not in {c["collection_id"] for c in db.list_collections(conn)}:
            raise KeyError(f"no collection {collection}")

        # 1-2. sources and files
        folders = [Path(f["path"]) for f in db.source_folders(conn, collection)]
        uploads = uploads_dir(settings, collection)
        present: list[Path] = []
        for folder in folders:
            if folder.is_dir():
                present.append(folder)
                emit("source", str(folder))
            else:
                report.missing_folders.append(str(folder))
                emit("missing-folder", str(folder), "not found now: its documents are kept")
        found, skipped = find_documents(present)
        if uploads.is_dir():
            uploaded, skipped_uploads = find_documents([uploads])
            found += [FoundFile(f.path, f"{UPLOAD_PREFIX}-{f.key}") for f in uploaded]
            skipped += skipped_uploads
        for path, why in skipped:
            report.skipped.append((str(path), why))
            emit("skipped", str(path), why)

        # 3. ingest what is new or changed
        parse_start = time.perf_counter()
        parsed_docs: list[str] = []
        known_before = {d.doc_id for d in db.list_documents(conn, collection_id=collection)}
        for item in found:
            check_cancel()
            try:
                [result] = ingestor.ingest_files([item])
            except (ValueError, ParseError, OSError) as exc:
                report.failed.append((str(item.path), str(exc)))
                emit("failed", str(item.path), str(exc))
                continue
            if result.status == "skipped":
                report.unchanged.append(result.doc_id)
                emit("unchanged", result.doc_id)
            elif result.status == "failed":
                report.failed.append((str(item.path), result.issues))
                emit("failed", str(item.path), result.issues)
            else:
                (report.updated if result.doc_id in known_before else report.added).append(result.doc_id)
                parsed_docs.append(result.doc_id)
                emit("updated" if result.doc_id in known_before else "added", result.doc_id, result.status)
        report.seconds["parse"] = time.perf_counter() - parse_start

        # 4. removals: documents from these folders whose file is gone
        found_paths = {str(item.path.resolve()) for item in found}
        roots = [*present, uploads]
        for document in db.list_documents(conn, collection_id=collection, source="file"):
            original = document.metadata.get("original_path")
            if not any(under(original, root) for root in roots) or original in found_paths:
                continue
            if Path(original).exists():
                continue  # still there, but skipped this time (now too large, say): keep it
            check_cancel()
            db.delete_document(conn, document.doc_id)
            if document.source_uri and under(document.source_uri, settings.data_dir / "raw"):
                Path(document.source_uri).unlink(missing_ok=True)
            report.removed.append(document.doc_id)
            emit("removed", document.doc_id, "its file is gone")

        # 5. chunk the re-parsed documents
        chunk_start = time.perf_counter()
        if parsed_docs:
            if counter is None:
                from rag.chunk.tokens import HFTokenCounter

                counter = HFTokenCounter(settings.embedding_model)
            chunker = Chunker(counter, settings.chunk_max_tokens)
            for doc_id in parsed_docs:
                document = db.get_document(conn, doc_id)
                if document is None or document.parse_status not in ("ok", "suspect"):
                    continue
                chunks = chunker.chunk_document(db.get_sections(conn, doc_id))
                db.replace_chunks(conn, doc_id, chunks)
                report.n_chunks += len(chunks)
                emit("chunked", doc_id, f"{len(chunks)} chunks")
        report.seconds["chunk"] = time.perf_counter() - chunk_start

        # 6. a new dense build if anything changed
        index_start = time.perf_counter()
        if report.changed or rebuild:
            check_cancel()
            from rag.index.build import EmbeddingCache, build_dense_index

            if counter is None:
                from rag.chunk.tokens import HFTokenCounter

                counter = HFTokenCounter(settings.embedding_model)
            if embedder is None:
                from rag.embed.embedder import Embedder

                embedder = Embedder(
                    settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size
                )
            indexes_dir = settings.data_dir / "indexes" / "dense"
            if db.count_chunks(conn) == 0:  # everything was removed: no index rather than an empty one
                (indexes_dir / "CURRENT").unlink(missing_ok=True)
                emit("indexed", "none", "no chunks left in the database: no index")
                report.seconds["index"] = time.perf_counter() - index_start
                for folder in present:
                    db.mark_folder_synced(conn, collection, str(folder))
                report.seconds["total"] = time.perf_counter() - started
                return report
            cache = EmbeddingCache(
                settings.data_dir / "embeddings" / f"{settings.embedding_model.replace('/', '__')}.npz"
            )
            built = build_dense_index(conn, embedder, counter, indexes_dir, cache)
            report.build_id, report.n_embedded = built.build_id, built.n_embedded
            pruned = prune_builds(indexes_dir)
            emit("indexed", built.build_id, f"{built.n_chunks} chunks, {built.n_embedded} newly embedded")
            if pruned:
                emit("pruned", ", ".join(pruned), f"old builds deleted (keeping {KEEP_BUILDS})")
        report.seconds["index"] = time.perf_counter() - index_start

        for folder in present:
            db.mark_folder_synced(conn, collection, str(folder))
    finally:
        ingestor.close()
    report.seconds["total"] = time.perf_counter() - started
    return report


def print_report(report: SyncReport) -> None:
    print(
        f"\n{report.collection}: {len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.unchanged)} unchanged, {len(report.removed)} removed, {len(report.failed)} failed, "
        f"{len(report.skipped)} skipped"
    )
    for path, why in report.failed:
        print(f"  failed   {path}: {why}")
    for path in report.missing_folders:
        print(f"  missing  {path} (its documents are kept)")
    if report.build_id:
        print(f"  index    {report.build_id} ({report.n_embedded} newly embedded) is CURRENT")
    elif not report.changed:
        print("  index    unchanged (nothing to rebuild)")
    print("  seconds  " + ", ".join(f"{k} {v:.1f}" for k, v in report.seconds.items()))


def run(parsed_args) -> int:
    settings = get_settings()
    conn = db.open_db(settings.db_path)
    collection = parsed_args.collection
    try:
        if parsed_args.add_folder:
            folder = parsed_args.add_folder.expanduser().resolve()
            if not folder.is_dir():
                print(f"error: {folder} is not a folder")
                return 1
            db.ensure_collection(conn, collection)
            added = db.add_source_folder(conn, collection, str(folder))
            print(f"{'added' if added else 'already registered'}: {folder} -> {collection}")
        if parsed_args.remove_folder:
            folder = parsed_args.remove_folder.expanduser().resolve()
            if not db.remove_source_folder(conn, collection, str(folder)):
                print(f"error: {folder} is not a folder of {collection}")
                return 1
            removed = 0
            for document in db.list_documents(conn, collection_id=collection, source="file"):
                if under(document.metadata.get("original_path"), folder):
                    db.delete_document(conn, document.doc_id)
                    if document.source_uri and under(document.source_uri, settings.data_dir / "raw"):
                        Path(document.source_uri).unlink(missing_ok=True)
                    removed += 1
            print(f"removed {folder} from {collection} and its {removed} documents")
    finally:
        conn.close()

    def show(event: SyncEvent) -> None:
        if event.kind in ("added", "updated", "removed", "failed", "missing-folder", "indexed"):
            print(f"{event.kind:9} {event.name}" + (f"  ({event.detail})" if event.detail else ""), flush=True)

    try:
        report = sync_collection(settings, collection, on_event=show, rebuild=bool(parsed_args.remove_folder))
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return 1
    print_report(report)
    return 1 if report.failed else 0


def main() -> int:
    from rag.main import run_module

    return run_module(["sync"])


if __name__ == "__main__":
    sys.exit(main())
