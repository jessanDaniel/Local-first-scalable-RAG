import sqlite3
from datetime import date

import pytest

from rag.config import Settings
from rag.models import Chunk, Paper, Section, make_chunk_id
from rag.store import db

SECTION_TEXT = "LoRA freezes the pretrained weights.\n\nIt trains low-rank matrices A and B."


def make_paper(**overrides) -> Paper:
    fields = {
        "arxiv_id": "2106.09685",
        "version": 2,
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen"],
        "abstract": "We propose Low-Rank Adaptation.",
        "categories": ["cs.CL", "cs.LG"],
        "primary_category": "cs.CL",
        "published": date(2021, 6, 17),
    }
    fields.update(overrides)
    return Paper(**fields)


def make_section(ord: int = 0, text: str = SECTION_TEXT) -> Section:
    return Section(
        arxiv_id="2106.09685", ord=ord, heading="4 Method", level=1,
        path="4 Method", text=text, anchor=f"S{ord}",
    )


def make_chunk(text: str, ord: int = 0, section_ord: int = 0, section_text: str = SECTION_TEXT) -> Chunk:
    start = section_text.index(text)
    return Chunk(
        arxiv_id="2106.09685", section_ord=section_ord, ord=ord, chunk_type="body",
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


def test_paper_round_trip_and_upsert(conn):
    db.upsert_paper(conn, make_paper())
    db.upsert_paper(conn, make_paper(version=3, source_format="html", parser="arxiv_html"))
    paper = db.get_paper(conn, "2106.09685")
    assert paper == make_paper(version=3, source_format="html", parser="arxiv_html")
    assert paper.url == "https://arxiv.org/abs/2106.09685"
    assert db.get_paper(conn, "0000.00000") is None


def test_parse_status_filter(conn):
    db.upsert_paper(conn, make_paper())
    db.upsert_paper(conn, make_paper(arxiv_id="2305.14314", title="QLoRA"))
    db.set_parse_status(conn, "2305.14314", "failed", error="no HTML")
    assert [p.arxiv_id for p in db.list_papers(conn, "pending")] == ["2106.09685"]
    failed = db.list_papers(conn, "failed")
    assert [(p.arxiv_id, p.parse_error) for p in failed] == [("2305.14314", "no HTML")]


def test_set_parse_status_unknown_paper(conn):
    with pytest.raises(KeyError):
        db.set_parse_status(conn, "0000.00000", "ok")


def test_chunk_offsets_point_into_section_text(conn):
    db.upsert_paper(conn, make_paper())
    db.replace_sections(conn, "2106.09685", [make_section()])
    chunks = [
        make_chunk("LoRA freezes the pretrained weights.", ord=0),
        make_chunk("It trains low-rank matrices A and B.", ord=1),
    ]
    db.replace_chunks(conn, "2106.09685", chunks)

    section = db.get_sections(conn, "2106.09685")[0]
    for chunk in db.get_chunks(conn, "2106.09685"):
        assert section.text[chunk.char_start:chunk.char_end] == chunk.text


def test_replace_is_idempotent_and_reparse_drops_chunks(conn):
    db.upsert_paper(conn, make_paper())
    db.replace_sections(conn, "2106.09685", [make_section()])
    chunk = make_chunk("LoRA freezes the pretrained weights.")
    db.replace_chunks(conn, "2106.09685", [chunk])
    db.replace_chunks(conn, "2106.09685", [chunk])
    assert db.count_chunks(conn) == 1

    db.replace_sections(conn, "2106.09685", [make_section()])
    assert db.count_chunks(conn) == 0


def test_get_chunks_by_ids_keeps_rank_order(conn):
    db.upsert_paper(conn, make_paper())
    db.replace_sections(conn, "2106.09685", [make_section()])
    first = make_chunk("LoRA freezes the pretrained weights.", ord=0)
    second = make_chunk("It trains low-rank matrices A and B.", ord=1)
    db.replace_chunks(conn, "2106.09685", [first, second])

    found = db.get_chunks_by_ids(conn, [second.chunk_id, "missing", first.chunk_id])
    assert [c.chunk_id for c in found] == [second.chunk_id, first.chunk_id]
    assert db.get_chunks_by_ids(conn, []) == []


def test_chunk_for_unknown_section_is_rejected(conn):
    db.upsert_paper(conn, make_paper())
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_chunks(conn, "2106.09685", [make_chunk("LoRA freezes the pretrained weights.")])


def test_duplicate_chunk_text_is_rejected(conn):
    db.upsert_paper(conn, make_paper())
    db.replace_sections(conn, "2106.09685", [make_section()])
    text = "LoRA freezes the pretrained weights."
    with pytest.raises(ValueError, match="duplicate"):
        db.replace_chunks(conn, "2106.09685", [make_chunk(text, ord=0), make_chunk(text, ord=1)])


def test_bad_chunk_type_is_rejected_by_schema(conn):
    db.upsert_paper(conn, make_paper())
    db.replace_sections(conn, "2106.09685", [make_section()])
    chunk = make_chunk("LoRA freezes the pretrained weights.")
    chunk.chunk_type = "footnote"
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_chunks(conn, "2106.09685", [chunk])


def test_deleting_a_paper_cascades(conn):
    db.upsert_paper(conn, make_paper())
    db.replace_sections(conn, "2106.09685", [make_section()])
    db.replace_chunks(conn, "2106.09685", [make_chunk("LoRA freezes the pretrained weights.")])
    with conn:
        conn.execute("DELETE FROM papers WHERE arxiv_id = '2106.09685'")
    assert db.count_chunks(conn) == 0
    assert db.get_sections(conn, "2106.09685") == []


def test_chunk_id_depends_on_content_not_position():
    assert make_chunk_id("2106.09685", 0, "same text") == make_chunk_id("2106.09685", 0, "same text")
    assert make_chunk_id("2106.09685", 0, "same text") != make_chunk_id("2106.09685", 1, "same text")
    moved = make_chunk("LoRA freezes the pretrained weights.", ord=5)
    assert moved.chunk_id == make_chunk("LoRA freezes the pretrained weights.", ord=0).chunk_id


def test_bad_offsets_are_rejected():
    with pytest.raises(ValueError, match="bad offsets"):
        Chunk(
            arxiv_id="x", section_ord=0, ord=0, chunk_type="body", text="t",
            char_start=5, char_end=5, n_tokens=1, chunker_version="test-1",
        )


def test_db_path_follows_data_dir(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.db_path == tmp_path / "rag.db"
