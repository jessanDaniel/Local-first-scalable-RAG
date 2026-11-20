-- Schema version 3, exactly as created before collections (commit 914b100). Used to test the v3 -> v4 migration.
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
