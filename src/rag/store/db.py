"""SQLite store: the source of truth for papers, sections and chunks.

Search indexes (FAISS, Tantivy) only hold chunk ids. Text, metadata and
citation data always come from here, so any index can be deleted and rebuilt.

Writes for one paper replace that paper's previous rows inside one
transaction, so re-running parse or chunk on a paper is safe.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

from rag.models import Chunk, Paper, ParseStatus, Section

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    version           INTEGER NOT NULL,
    title             TEXT NOT NULL,
    authors_json      TEXT NOT NULL,
    abstract          TEXT NOT NULL,
    categories_json   TEXT NOT NULL,
    primary_category  TEXT NOT NULL,
    published         TEXT NOT NULL,
    updated           TEXT,
    source_format     TEXT CHECK (source_format IN ('html', 'pdf')),
    source_sha256     TEXT,
    parser            TEXT,
    parse_status      TEXT NOT NULL DEFAULT 'pending'
                      CHECK (parse_status IN ('pending', 'ok', 'suspect', 'failed')),
    parse_error       TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sections (
    section_id  TEXT PRIMARY KEY,
    arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    heading     TEXT NOT NULL,
    level       INTEGER NOT NULL,
    path        TEXT NOT NULL,
    text        TEXT NOT NULL,
    anchor      TEXT,
    page_start  INTEGER,
    UNIQUE (arxiv_id, ord)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    arxiv_id         TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
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
    UNIQUE (arxiv_id, ord),
    CHECK (char_start >= 0 AND char_end > char_start)
);

CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(parse_status);
CREATE INDEX IF NOT EXISTS idx_sections_paper ON sections(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(arxiv_id);
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


def init_schema(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current not in (0, SCHEMA_VERSION):
        raise SchemaVersionError(
            f"database schema is version {current}, code expects {SCHEMA_VERSION}; "
            "rebuild the database or add a migration"
        )
    with conn:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    init_schema(conn)
    return conn


# --- papers -----------------------------------------------------------------


def upsert_paper(conn: sqlite3.Connection, paper: Paper) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, version, title, authors_json, abstract, categories_json,
                primary_category, published, updated, source_format, source_sha256,
                parser, parse_status, parse_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (arxiv_id) DO UPDATE SET
                version = excluded.version,
                title = excluded.title,
                authors_json = excluded.authors_json,
                abstract = excluded.abstract,
                categories_json = excluded.categories_json,
                primary_category = excluded.primary_category,
                published = excluded.published,
                updated = excluded.updated,
                source_format = excluded.source_format,
                source_sha256 = excluded.source_sha256,
                parser = excluded.parser,
                parse_status = excluded.parse_status,
                parse_error = excluded.parse_error,
                updated_at = datetime('now')
            """,
            (
                paper.arxiv_id,
                paper.version,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                json.dumps(paper.categories),
                paper.primary_category,
                paper.published.isoformat(),
                paper.updated.isoformat() if paper.updated else None,
                paper.source_format,
                paper.source_sha256,
                paper.parser,
                paper.parse_status,
                paper.parse_error,
            ),
        )


def set_parse_status(
    conn: sqlite3.Connection,
    arxiv_id: str,
    status: ParseStatus,
    error: str | None = None,
) -> None:
    with conn:
        cursor = conn.execute(
            """
            UPDATE papers SET parse_status = ?, parse_error = ?, updated_at = datetime('now')
            WHERE arxiv_id = ?
            """,
            (status, error, arxiv_id),
        )
    if cursor.rowcount == 0:
        raise KeyError(f"no paper {arxiv_id}")


def get_paper(conn: sqlite3.Connection, arxiv_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return _paper_from_row(row) if row else None


def list_papers(conn: sqlite3.Connection, parse_status: ParseStatus | None = None) -> list[Paper]:
    if parse_status is None:
        rows = conn.execute("SELECT * FROM papers ORDER BY arxiv_id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM papers WHERE parse_status = ? ORDER BY arxiv_id", (parse_status,)
        ).fetchall()
    return [_paper_from_row(row) for row in rows]


def _paper_from_row(row: sqlite3.Row) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"],
        version=row["version"],
        title=row["title"],
        authors=json.loads(row["authors_json"]),
        abstract=row["abstract"],
        categories=json.loads(row["categories_json"]),
        primary_category=row["primary_category"],
        published=date.fromisoformat(row["published"]),
        updated=date.fromisoformat(row["updated"]) if row["updated"] else None,
        source_format=row["source_format"],
        source_sha256=row["source_sha256"],
        parser=row["parser"],
        parse_status=row["parse_status"],
        parse_error=row["parse_error"],
    )


# --- sections ---------------------------------------------------------------


def replace_sections(conn: sqlite3.Connection, arxiv_id: str, sections: list[Section]) -> None:
    """Replace a paper's sections. Its chunks are deleted too (they point at the old sections)."""
    for section in sections:
        if section.arxiv_id != arxiv_id:
            raise ValueError(f"section {section.section_id} does not belong to {arxiv_id}")
    with conn:
        conn.execute("DELETE FROM chunks WHERE arxiv_id = ?", (arxiv_id,))
        conn.execute("DELETE FROM sections WHERE arxiv_id = ?", (arxiv_id,))
        conn.executemany(
            """
            INSERT INTO sections (section_id, arxiv_id, ord, heading, level, path, text, anchor, page_start)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (s.section_id, s.arxiv_id, s.ord, s.heading, s.level, s.path, s.text, s.anchor, s.page_start)
                for s in sections
            ],
        )


def get_sections(conn: sqlite3.Connection, arxiv_id: str) -> list[Section]:
    rows = conn.execute(
        "SELECT * FROM sections WHERE arxiv_id = ? ORDER BY ord", (arxiv_id,)
    ).fetchall()
    return [
        Section(
            arxiv_id=row["arxiv_id"],
            ord=row["ord"],
            heading=row["heading"],
            level=row["level"],
            path=row["path"],
            text=row["text"],
            anchor=row["anchor"],
            page_start=row["page_start"],
        )
        for row in rows
    ]


# --- chunks -----------------------------------------------------------------


def replace_chunks(conn: sqlite3.Connection, arxiv_id: str, chunks: list[Chunk]) -> None:
    """Replace a paper's chunks. Two chunks with the same text in the same section is an error:
    they would get the same chunk_id, and the chunker should not emit duplicates."""
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.arxiv_id != arxiv_id:
            raise ValueError(f"chunk {chunk.chunk_id} does not belong to {arxiv_id}")
        if chunk.chunk_id in seen:
            raise ValueError(
                f"duplicate chunk text in section {chunk.section_id} (chunk_id {chunk.chunk_id})"
            )
        seen.add(chunk.chunk_id)
    with conn:
        conn.execute("DELETE FROM chunks WHERE arxiv_id = ?", (arxiv_id,))
        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, arxiv_id, section_id, section_ord, ord, chunk_type, text,
                char_start, char_end, n_tokens, chunker_version, page_start, page_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.chunk_id, c.arxiv_id, c.section_id, c.section_ord, c.ord, c.chunk_type, c.text,
                    c.char_start, c.char_end, c.n_tokens, c.chunker_version, c.page_start, c.page_end,
                )
                for c in chunks
            ],
        )


def get_chunks(conn: sqlite3.Connection, arxiv_id: str) -> list[Chunk]:
    rows = conn.execute(
        "SELECT * FROM chunks WHERE arxiv_id = ? ORDER BY ord", (arxiv_id,)
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
        arxiv_id=row["arxiv_id"],
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
