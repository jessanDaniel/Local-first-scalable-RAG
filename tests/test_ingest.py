from datetime import date
from pathlib import Path

import pymupdf
import pytest

from rag.ingest import ingest_paper
from rag.models import Paper, Section
from rag.parse.base import ParsedPaper, ParseError, Parser, ParserUnavailableError
from rag.parse.pymupdf import FALLBACK_ISSUE, PyMuPDFParser
from rag.store import db


def make_paper() -> Paper:
    return Paper(
        arxiv_id="1412.6980", version=9, title="Adam", authors=["Kingma", "Ba"],
        abstract="We introduce Adam.", categories=["cs.LG"], primary_category="cs.LG",
        published=date(2014, 12, 22), source_format="pdf",
    )


def write_pdf(path: Path, pages: list[str]) -> Path:
    document = pymupdf.open()
    for text in pages:
        document.new_page().insert_text((72, 72), text)
    document.save(path)
    return path


class FakeClient:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def download_pdf(self, paper, pdf_dir):
        return self.pdf_path, "sha", True


class FakeParser(Parser):
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error

    def parse(self, pdf_path, arxiv_id):
        if self.error:
            raise self.error
        return self.result


def good_parse(arxiv_id: str) -> ParsedPaper:
    words = " ".join(["word"] * 200)
    sections = [
        Section.from_paragraphs(arxiv_id, i, f"{i} S", 1, f"{i} S", [(words, i + 1, i + 1)])
        for i in range(2)
    ]
    # No title or abstract from the PDF: arXiv metadata covers those, so status stays ok.
    paper = ParsedPaper(arxiv_id, "grobid-test", "", [], "", sections, 2, 5)
    paper.issues = ["no title", "no abstract"]
    return paper


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "rag.db")
    yield connection
    connection.close()


def run(conn, tmp_path, primary, fallback=None):
    pdf = write_pdf(tmp_path / "paper.pdf", ["Some body text on page one.", "More text. References"])
    paper = make_paper()
    db.upsert_paper(conn, paper)
    return ingest_paper(
        paper, conn, FakeClient(pdf), primary, fallback or PyMuPDFParser(),
        tmp_path / "pdf", tmp_path / "parsed",
    )


def test_grobid_success_is_stored_as_ok(conn, tmp_path):
    result = run(conn, tmp_path, FakeParser("grobid", result=good_parse("1412.6980")))
    assert (result.status, result.used_fallback, result.n_sections) == ("ok", False, 2)
    stored = db.get_paper(conn, "1412.6980")
    assert (stored.parse_status, stored.parser, stored.title) == ("ok", "grobid-test", "Adam")
    assert len(db.get_sections(conn, "1412.6980")) == 2
    assert (tmp_path / "parsed" / "1412.6980.json").exists()


def test_grobid_failure_falls_back_to_pymupdf(conn, tmp_path):
    result = run(conn, tmp_path, FakeParser("grobid", error=ParseError("HTTP 500")))
    assert result.used_fallback and result.status == "suspect"
    assert result.parser.startswith("pymupdf")
    stored = db.get_paper(conn, "1412.6980")
    assert FALLBACK_ISSUE in stored.parse_error and "HTTP 500" in stored.parse_error
    sections = db.get_sections(conn, "1412.6980")
    assert [s.path for s in sections] == ["Page 1", "Page 2"]
    assert sections[1].paragraphs[0].page_start == 2


def test_both_parsers_failing_marks_failed(conn, tmp_path):
    result = run(
        conn, tmp_path,
        FakeParser("grobid", error=ParseError("HTTP 500")),
        FakeParser("pymupdf", error=ParseError("no text")),
    )
    assert result.status == "failed"
    stored = db.get_paper(conn, "1412.6980")
    assert stored.parse_status == "failed" and "no text" in stored.parse_error


def test_grobid_down_stops_instead_of_falling_back(conn, tmp_path):
    with pytest.raises(ParserUnavailableError):
        run(conn, tmp_path, FakeParser("grobid", error=ParserUnavailableError("down")))
    assert db.get_paper(conn, "1412.6980").parse_status == "pending"


def test_pymupdf_stops_at_references(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", ["Intro text.", "Method text.", "References", "[1] A paper."])
    parsed = PyMuPDFParser().parse(pdf, "x")
    assert [s.path for s in parsed.sections] == ["Page 1", "Page 2"]
    assert parsed.n_pages == 4
