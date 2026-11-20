"""SQLite store: the source of truth for documents, sections and chunks.

Search indexes (FAISS, Tantivy) only hold chunk ids. Text, metadata and
citation data always come from here, so any index can be deleted and rebuilt.

Writes for one document replace that document's previous rows inside one
transaction, so re-running parse or chunk on a document is safe.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

from rag.models import Chunk, Document, Paragraph, ParseStatus, Section

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,
    source            TEXT NOT NULL CHECK (source IN ('arxiv', 'file')),
    title             TEXT NOT NULL,
    authors_json      TEXT NOT NULL DEFAULT '[]',
    abstract          TEXT NOT NULL DEFAULT '',
    published         TEXT,
    source_uri        TEXT,
    source_format     TEXT CHECK (source_format IN ('html', 'pdf')),
    source_sha256     TEXT,
    metadata_json     TEXT NOT NULL DEFAULT '{}',
    parser            TEXT,
    parse_status      TEXT NOT NULL DEFAULT 'pending'
                      CHECK (parse_status IN ('pending', 'ok', 'suspect', 'failed')),
    parse_error       TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (doc_id LIKE source || ':_%')
);

CREATE TABLE IF NOT EXISTS sections (
    section_id  TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    heading     TEXT NOT NULL,
    level       INTEGER NOT NULL,
    path        TEXT NOT NULL,
    text        TEXT NOT NULL,
    paragraphs_json TEXT NOT NULL DEFAULT '[]',
    anchor      TEXT,
    page_start  INTEGER,
    UNIQUE (doc_id, ord)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    doc_id           TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id       TEXT NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    section_ord      INTEGER NOT NULL,
    ord              INTEGER NOT NULL,
    chunk_type       TEXT NOT NULL CHECK (chunk_type IN ('abstract', 'body', 'caption', 'table')),
    text             TEXT NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    n_tokens         INTEGER NOT NULL,
    chunker_version  TEXT NOT NULL,
    page_start       INTEGER,
    page_end         INTEGER,
    UNIQUE (doc_id, ord),
    CHECK (char_start >= 0 AND char_end > char_start)
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(parse_status);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);

CREATE TABLE IF NOT EXISTS index_builds (
    build_id         TEXT PRIMARY KEY,
    kind             TEXT NOT NULL CHECK (kind IN ('dense', 'sparse')),
    embedding_model  TEXT,
    chunker_version  TEXT NOT NULL,
    params_json      TEXT NOT NULL,
    n_chunks         INTEGER NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('building', 'ready', 'failed')),
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SchemaVersionError(RuntimeError):
    """The database was created by a different schema version than this code expects."""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# Each migration takes a database from version N to N + 1. Version 2 -> 3 (papers -> documents,
# arxiv_id -> doc_id) has none on purpose: it was made while the database held only re-ingestable
# data and no chunks or indexes, so the database is rebuilt instead (delete it, run ingest again).
MIGRATIONS = {
    1: ["ALTER TABLE sections ADD COLUMN paragraphs_json TEXT NOT NULL DEFAULT '[]'"],
}


def init_schema(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"database schema is version {current}, code expects {SCHEMA_VERSION}; "
            "this code is older than the database"
        )
    missing = [version for version in range(current, SCHEMA_VERSION) if current and version not in MIGRATIONS]
    if missing:
        raise SchemaVersionError(
            f"database schema is version {current}, code expects {SCHEMA_VERSION}, and there is no "
            f"migration from version {missing[0]}; delete the database file and run ingest again"
        )
    with conn:
        if current == 0:
            conn.executescript(SCHEMA)
        else:
            for version in range(current, SCHEMA_VERSION):
                for statement in MIGRATIONS[version]:
                    conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    init_schema(conn)
    return conn


# --- documents --------------------------------------------------------------


def upsert_document(conn: sqlite3.Connection, document: Document) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO documents (
                doc_id, source, title, authors_json, abstract, published, source_uri,
                source_format, source_sha256, metadata_json, parser, parse_status, parse_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id) DO UPDATE SET
                source = excluded.source,
                title = excluded.title,
                authors_json = excluded.authors_json,
                abstract = excluded.abstract,
                published = excluded.published,
                source_uri = excluded.source_uri,
                source_format = excluded.source_format,
                source_sha256 = excluded.source_sha256,
                metadata_json = excluded.metadata_json,
                parser = excluded.parser,
                parse_status = excluded.parse_status,
                parse_error = excluded.parse_error,
                updated_at = datetime('now')
            """,
            (
                document.doc_id,
                document.source,
                document.title,
                json.dumps(document.authors),
                document.abstract,
                document.published.isoformat() if document.published else None,
                document.source_uri,
                document.source_format,
                document.source_sha256,
                json.dumps(document.metadata, sort_keys=True),
                document.parser,
                document.parse_status,
                document.parse_error,
            ),
        )


def set_parse_status(
    conn: sqlite3.Connection,
    doc_id: str,
    status: ParseStatus,
    error: str | None = None,
) -> None:
    with conn:
        cursor = conn.execute(
            """
            UPDATE documents SET parse_status = ?, parse_error = ?, updated_at = datetime('now')
            WHERE doc_id = ?
            """,
            (status, error, doc_id),
        )
    if cursor.rowcount == 0:
        raise KeyError(f"no document {doc_id}")


def get_document(conn: sqlite3.Connection, doc_id: str) -> Document | None:
    row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return _document_from_row(row) if row else None


def list_documents(
    conn: sqlite3.Connection,
    parse_status: ParseStatus | None = None,
    source: str | None = None,
) -> list[Document]:
    query = "SELECT * FROM documents"
    conditions, params = [], []
    if parse_status is not None:
        conditions.append("parse_status = ?")
        params.append(parse_status)
    if source is not None:
        conditions.append("source = ?")
        params.append(source)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    rows = conn.execute(query + " ORDER BY doc_id", params).fetchall()
    return [_document_from_row(row) for row in rows]


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        source=row["source"],
        title=row["title"],
        authors=json.loads(row["authors_json"]),
        abstract=row["abstract"],
        published=date.fromisoformat(row["published"]) if row["published"] else None,
        source_uri=row["source_uri"],
        source_format=row["source_format"],
        source_sha256=row["source_sha256"],
        metadata=json.loads(row["metadata_json"]),
        parser=row["parser"],
        parse_status=row["parse_status"],
        parse_error=row["parse_error"],
    )


# --- sections ---------------------------------------------------------------


def replace_sections(conn: sqlite3.Connection, doc_id: str, sections: list[Section]) -> None:
    """Replace a document's sections. Its chunks are deleted too (they point at the old sections)."""
    for section in sections:
        if section.doc_id != doc_id:
            raise ValueError(f"section {section.section_id} does not belong to {doc_id}")
    with conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            """
            INSERT INTO sections (
                section_id, doc_id, ord, heading, level, path, text, paragraphs_json, anchor, page_start
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.section_id, s.doc_id, s.ord, s.heading, s.level, s.path, s.text,
                    json.dumps([[p.char_start, p.char_end, p.page_start, p.page_end] for p in s.paragraphs]),
                    s.anchor, s.page_start,
                )
                for s in sections
            ],
        )


def get_sections(conn: sqlite3.Connection, doc_id: str) -> list[Section]:
    rows = conn.execute(
        "SELECT * FROM sections WHERE doc_id = ? ORDER BY ord", (doc_id,)
    ).fetchall()
    return [
        Section(
            doc_id=row["doc_id"],
            ord=row["ord"],
            heading=row["heading"],
            level=row["level"],
            path=row["path"],
            text=row["text"],
            paragraphs=[Paragraph(*span) for span in json.loads(row["paragraphs_json"])],
            anchor=row["anchor"],
            page_start=row["page_start"],
        )
        for row in rows
    ]


# --- chunks -----------------------------------------------------------------


def replace_chunks(conn: sqlite3.Connection, doc_id: str, chunks: list[Chunk]) -> None:
    """Replace a document's chunks. Two chunks with the same text in the same section is an error:
    they would get the same chunk_id, and the chunker should not emit duplicates."""
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id != doc_id:
            raise ValueError(f"chunk {chunk.chunk_id} does not belong to {doc_id}")
        if chunk.chunk_id in seen:
            raise ValueError(
                f"duplicate chunk text in section {chunk.section_id} (chunk_id {chunk.chunk_id})"
            )
        seen.add(chunk.chunk_id)
    with conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, section_id, section_ord, ord, chunk_type, text,
                char_start, char_end, n_tokens, chunker_version, page_start, page_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.chunk_id, c.doc_id, c.section_id, c.section_ord, c.ord, c.chunk_type, c.text,
                    c.char_start, c.char_end, c.n_tokens, c.chunker_version, c.page_start, c.page_end,
                )
                for c in chunks
            ],
        )


def get_chunks(conn: sqlite3.Connection, doc_id: str) -> list[Chunk]:
    rows = conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,)
    ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def get_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[Chunk]:
    """Look up chunks by id, returned in the order the ids were given (retrieval rank order).
    Ids that are not in the database are skipped."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
    ).fetchall()
    by_id = {row["chunk_id"]: _chunk_from_row(row) for row in rows}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    chunk = Chunk(
        doc_id=row["doc_id"],
        section_ord=row["section_ord"],
        ord=row["ord"],
        chunk_type=row["chunk_type"],
        text=row["text"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        n_tokens=row["n_tokens"],
        chunker_version=row["chunker_version"],
        page_start=row["page_start"],
        page_end=row["page_end"],
    )
    if chunk.chunk_id != row["chunk_id"]:
        raise ValueError(f"stored chunk_id {row['chunk_id']} does not match its content")
    return chunk


# --- index builds -----------------------------------------------------------


def record_index_build(
    conn: sqlite3.Connection,
    build_id: str,
    kind: str,
    embedding_model: str | None,
    chunker_version: str,
    params: dict,
    n_chunks: int,
    status: str = "building",
) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO index_builds (build_id, kind, embedding_model, chunker_version, params_json, n_chunks, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (build_id, kind, embedding_model, chunker_version, json.dumps(params, sort_keys=True), n_chunks, status),
        )


def set_index_build_status(conn: sqlite3.Connection, build_id: str, status: str) -> None:
    with conn:
        conn.execute("UPDATE index_builds SET status = ? WHERE build_id = ?", (status, build_id))


def list_index_builds(conn: sqlite3.Connection, kind: str | None = None) -> list[dict]:
    query = "SELECT * FROM index_builds" + (" WHERE kind = ?" if kind else "") + " ORDER BY created_at, build_id"
    rows = conn.execute(query, (kind,) if kind else ()).fetchall()
    return [dict(row) | {"params": json.loads(row["params_json"])} for row in rows]


def all_chunk_ids(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT chunk_id FROM chunks ORDER BY doc_id, ord")]


# --- status -----------------------------------------------------------------


def stats(conn: sqlite3.Connection) -> dict:
    """Counts for `rag status`: documents by parse status and source, sections, chunks."""

    def grouped(sql: str) -> dict[str, int]:
        return {row[0]: row[1] for row in conn.execute(sql)}

    return {
        "schema_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "documents_by_status": grouped("SELECT parse_status, COUNT(*) FROM documents GROUP BY parse_status"),
        "documents_by_source": grouped("SELECT source, COUNT(*) FROM documents GROUP BY source"),
        "sections": conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
        "chunks": count_chunks(conn),
        "chunks_by_chunker": grouped("SELECT chunker_version, COUNT(*) FROM chunks GROUP BY chunker_version"),
        # Parsed documents with sections but no chunks: chunking has not been run since the parse.
        "unchunked_documents": conn.execute(
            """
            SELECT COUNT(*) FROM documents d
            WHERE d.parse_status IN ('ok', 'suspect')
              AND EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.doc_id)
              AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)
            """
        ).fetchone()[0],
    }
