import sqlite3
from datetime import date

import pytest

from rag.config import Settings
from rag.models import Chunk, Document, Paragraph, Section, make_chunk_id, make_doc_id
from rag.store import db

SECTION_TEXT = "LoRA freezes the pretrained weights.\n\nIt trains low-rank matrices A and B."


DOC = "arxiv:2106.09685"


def make_document(**overrides) -> Document:
    fields = {
        "doc_id": DOC,
        "source": "arxiv",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen"],
        "abstract": "We propose Low-Rank Adaptation.",
        "published": date(2021, 6, 17),
        "source_uri": "https://arxiv.org/abs/2106.09685",
        "metadata": {"arxiv_id": "2106.09685", "version": 2, "categories": ["cs.CL", "cs.LG"]},
    }
    fields.update(overrides)
    return Document(**fields)


def make_section(ord: int = 0, text: str = SECTION_TEXT) -> Section:
    return Section(
        doc_id=DOC, ord=ord, heading="4 Method", level=1,
        path="4 Method", text=text, anchor=f"S{ord}",
    )


def make_chunk(text: str, ord: int = 0, section_ord: int = 0, section_text: str = SECTION_TEXT) -> Chunk:
    start = section_text.index(text)
    return Chunk(
        doc_id=DOC, section_ord=section_ord, ord=ord, chunk_type="body",
        text=text, char_start=start, char_end=start + len(text),
        n_tokens=len(text.split()), chunker_version="test-1",
    )


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "rag.db")
    yield connection
    connection.close()


def test_open_db_twice_is_safe(tmp_path):
    db.open_db(tmp_path / "rag.db").close()
    connection = db.open_db(tmp_path / "rag.db")
    assert connection.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_newer_schema_version_is_refused(tmp_path):
    path = tmp_path / "rag.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(db.SchemaVersionError):
        db.open_db(path)


def test_document_round_trip_and_upsert(conn):
    db.upsert_document(conn, make_document())
    db.upsert_document(conn, make_document(title="LoRA v3", source_format="pdf", parser="grobid-0.8.2"))
    document = db.get_document(conn, DOC)
    assert document == make_document(title="LoRA v3", source_format="pdf", parser="grobid-0.8.2")
    assert document.key == "2106.09685"
    assert document.metadata["categories"] == ["cs.CL", "cs.LG"]
    assert db.get_document(conn, "arxiv:0000.00000") is None


def test_status_and_source_filters(conn):
    db.upsert_document(conn, make_document())
    db.upsert_document(conn, make_document(doc_id="arxiv:2305.14314", title="QLoRA"))
    db.upsert_document(conn, Document(doc_id="file:notes", source="file", title="My notes"))
    db.set_parse_status(conn, "arxiv:2305.14314", "failed", error="GROBID HTTP 500")
    assert [d.doc_id for d in db.list_documents(conn, "pending")] == [DOC, "file:notes"]
    assert [d.doc_id for d in db.list_documents(conn, source="file")] == ["file:notes"]
    failed = db.list_documents(conn, "failed")
    assert [(d.doc_id, d.parse_error) for d in failed] == [("arxiv:2305.14314", "GROBID HTTP 500")]


def test_doc_id_must_match_source():
    with pytest.raises(ValueError, match="must look like"):
        Document(doc_id="2106.09685", source="arxiv", title="no prefix")
    with pytest.raises(ValueError, match="must look like"):
        Document(doc_id="file:x", source="arxiv", title="wrong prefix")


def test_make_doc_id():
    assert make_doc_id("arxiv", "2106.09685") == "arxiv:2106.09685"
    assert make_doc_id("file", "The Perceptron (1958)") == "file:the-perceptron-1958"
    with pytest.raises(ValueError):
        make_doc_id("file", "((()))")


def test_set_parse_status_unknown_document(conn):
    with pytest.raises(KeyError):
        db.set_parse_status(conn, "0000.00000", "ok")


def test_chunk_offsets_point_into_section_text(conn):
    db.upsert_document(conn, make_document())
    db.replace_sections(conn, DOC, [make_section()])
    chunks = [
        make_chunk("LoRA freezes the pretrained weights.", ord=0),
        make_chunk("It trains low-rank matrices A and B.", ord=1),
    ]
    db.replace_chunks(conn, DOC, chunks)

    section = db.get_sections(conn, DOC)[0]
    for chunk in db.get_chunks(conn, DOC):
        assert section.text[chunk.char_start:chunk.char_end] == chunk.text


def test_replace_is_idempotent_and_reparse_drops_chunks(conn):
    db.upsert_document(conn, make_document())
    db.replace_sections(conn, DOC, [make_section()])
    chunk = make_chunk("LoRA freezes the pretrained weights.")
    db.replace_chunks(conn, DOC, [chunk])
    db.replace_chunks(conn, DOC, [chunk])
    assert db.count_chunks(conn) == 1

    db.replace_sections(conn, DOC, [make_section()])
    assert db.count_chunks(conn) == 0


def test_get_chunks_by_ids_keeps_rank_order(conn):
    db.upsert_document(conn, make_document())
    db.replace_sections(conn, DOC, [make_section()])
    first = make_chunk("LoRA freezes the pretrained weights.", ord=0)
    second = make_chunk("It trains low-rank matrices A and B.", ord=1)
    db.replace_chunks(conn, DOC, [first, second])

    found = db.get_chunks_by_ids(conn, [second.chunk_id, "missing", first.chunk_id])
    assert [c.chunk_id for c in found] == [second.chunk_id, first.chunk_id]
    assert db.get_chunks_by_ids(conn, []) == []


def test_chunk_for_unknown_section_is_rejected(conn):
    db.upsert_document(conn, make_document())
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_chunks(conn, DOC, [make_chunk("LoRA freezes the pretrained weights.")])


def test_duplicate_chunk_text_is_rejected(conn):
    db.upsert_document(conn, make_document())
    db.replace_sections(conn, DOC, [make_section()])
    text = "LoRA freezes the pretrained weights."
    with pytest.raises(ValueError, match="duplicate"):
        db.replace_chunks(conn, DOC, [make_chunk(text, ord=0), make_chunk(text, ord=1)])


def test_bad_chunk_type_is_rejected_by_schema(conn):
    db.upsert_document(conn, make_document())
    db.replace_sections(conn, DOC, [make_section()])
    chunk = make_chunk("LoRA freezes the pretrained weights.")
    chunk.chunk_type = "footnote"
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_chunks(conn, DOC, [chunk])


def test_deleting_a_document_cascades(conn):
    db.upsert_document(conn, make_document())
    db.replace_sections(conn, DOC, [make_section()])
    db.replace_chunks(conn, DOC, [make_chunk("LoRA freezes the pretrained weights.")])
    with conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (DOC,))
    assert db.count_chunks(conn) == 0
    assert db.get_sections(conn, DOC) == []


def test_chunk_id_depends_on_content_not_position():
    assert make_chunk_id(DOC, 0, "same text") == make_chunk_id(DOC, 0, "same text")
    assert make_chunk_id(DOC, 0, "same text") != make_chunk_id(DOC, 1, "same text")
    moved = make_chunk("LoRA freezes the pretrained weights.", ord=5)
    assert moved.chunk_id == make_chunk("LoRA freezes the pretrained weights.", ord=0).chunk_id


def test_bad_offsets_are_rejected():
    with pytest.raises(ValueError, match="bad offsets"):
        Chunk(
            doc_id="x", section_ord=0, ord=0, chunk_type="body", text="t",
            char_start=5, char_end=5, n_tokens=1, chunker_version="test-1",
        )


def test_db_path_follows_data_dir(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.db_path == tmp_path / "rag.db"


def test_paragraph_spans_round_trip(conn):
    db.upsert_document(conn, make_document())
    section = Section.from_paragraphs(
        DOC, 0, "4 Method", 1, "4 Method",
        [("LoRA freezes the pretrained weights.", 4, 4), ("It trains low-rank matrices A and B.", 4, 5)],
    )
    db.replace_sections(conn, DOC, [section])
    stored = db.get_sections(conn, DOC)[0]
    assert stored == section
    assert stored.page_start == 4
    assert [(p.page_start, p.page_end) for p in stored.paragraphs] == [(4, 4), (4, 5)]


def test_paragraph_offsets_must_fit_the_text():
    with pytest.raises(ValueError, match="paragraph offsets"):
        Section("x", 0, "h", 1, "h", "short", paragraphs=[Paragraph(0, 50)])


def test_old_schema_without_migration_is_refused(tmp_path):
    path = tmp_path / "rag.db"
    raw = sqlite3.connect(path)
    raw.executescript("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY); PRAGMA user_version = 2;")
    raw.close()
    with pytest.raises(db.SchemaVersionError, match="delete the database file"):
        db.open_db(path)
