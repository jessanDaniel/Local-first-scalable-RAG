from dataclasses import replace
from datetime import date
from pathlib import Path

import pymupdf
import pytest

from rag.config import Settings
from rag.ingest import Ingestor, parse_and_store
from rag.models import Document, Section
from rag.parse.base import ParsedDocument, ParseError, Parser, ParserUnavailableError
from rag.parse.pymupdf import FALLBACK_ISSUE, PyMuPDFParser
from rag.store import db

DOC = "arxiv:1412.6980"


def make_document() -> Document:
    return Document(
        doc_id=DOC,
        source="arxiv",
        title="Adam",
        authors=["Kingma", "Ba"],
        abstract="We introduce Adam.",
        published=date(2014, 12, 22),
        source_format="pdf",
        metadata={"arxiv_id": "1412.6980", "version": 9},
    )


def write_pdf(path: Path, pages: list[str], title: str = "") -> Path:
    document = pymupdf.open()
    for text in pages:
        document.new_page().insert_text((72, 72), text)
    if title:
        document.set_metadata({"title": title})
    document.save(path)
    return path


class FakeParser(Parser):
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = 0

    def parse(self, pdf_path, doc_id):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result(doc_id) if callable(self.result) else self.result


def good_parse(doc_id: str, title: str = "", authors: list[str] | None = None) -> ParsedDocument:
    words = " ".join(["word"] * 200)
    sections = [Section.from_paragraphs(doc_id, i, f"{i} S", 1, f"{i} S", [(words, i + 1, i + 1)]) for i in range(2)]
    parsed = ParsedDocument(doc_id, "grobid-test", title, authors or [], "", sections, 2, 5)
    parsed.issues = [] if title else ["no title"]
    return parsed


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "rag.db")
    yield connection
    connection.close()


def run(conn, tmp_path, primary, fallback=None):
    pdf = write_pdf(tmp_path / "paper.pdf", ["Some body text on page one.", "More text. References"])
    document = make_document()
    db.upsert_document(conn, document)
    return parse_and_store(document, pdf, conn, primary, fallback or PyMuPDFParser(), tmp_path / "parsed")


def test_grobid_success_is_stored_as_ok(conn, tmp_path):
    result = run(conn, tmp_path, FakeParser("grobid", result=good_parse))
    assert (result.status, result.used_fallback, result.n_sections) == ("ok", False, 2)
    stored = db.get_document(conn, DOC)
    assert (stored.parse_status, stored.parser, stored.title) == ("ok", "grobid-test", "Adam")
    assert len(db.get_sections(conn, DOC)) == 2
    assert (tmp_path / "parsed" / "arxiv_1412.6980.json").exists()


def test_grobid_failure_falls_back_to_pymupdf(conn, tmp_path):
    result = run(conn, tmp_path, FakeParser("grobid", error=ParseError("HTTP 500")))
    assert result.used_fallback and result.status == "suspect"
    assert result.parser.startswith("pymupdf")
    stored = db.get_document(conn, DOC)
    assert FALLBACK_ISSUE in stored.parse_error and "HTTP 500" in stored.parse_error
    sections = db.get_sections(conn, DOC)
    assert [s.path for s in sections] == ["Page 1", "Page 2"]
    assert sections[1].paragraphs[0].page_start == 2


def test_both_parsers_failing_marks_failed(conn, tmp_path):
    result = run(
        conn,
        tmp_path,
        FakeParser("grobid", error=ParseError("HTTP 500")),
        FakeParser("pymupdf", error=ParseError("no text")),
    )
    assert result.status == "failed"
    stored = db.get_document(conn, DOC)
    assert stored.parse_status == "failed" and "no text" in stored.parse_error


def test_grobid_down_stops_instead_of_falling_back(conn, tmp_path):
    with pytest.raises(ParserUnavailableError):
        run(conn, tmp_path, FakeParser("grobid", error=ParserUnavailableError("down")))
    assert db.get_document(conn, DOC).parse_status == "pending"


def test_pymupdf_stops_at_references(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", ["Intro text.", "Method text.", "References", "[1] A paper."])
    parsed = PyMuPDFParser().parse(pdf, "file:p")
    assert [s.path for s in parsed.sections] == ["Page 1", "Page 2"]
    assert parsed.n_pages == 4


def test_pymupdf_reads_pdf_title(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", ["Body."], title="The  Perceptron")
    assert PyMuPDFParser().parse(pdf, "file:p").title == "The Perceptron"


# --- local files ----------------------------------------------------------------


@pytest.fixture
def ingestor(tmp_path):
    ing = Ingestor(Settings(_env_file=None, data_dir=tmp_path / "data"))
    ing._checked_grobid = True  # no GROBID in tests
    ing.primary = FakeParser("grobid", result=lambda doc_id: good_parse(doc_id, "The Perceptron", ["Frank Rosenblatt"]))
    yield ing
    ing.close()


def test_local_file_is_copied_and_titled_from_the_parse(ingestor, tmp_path):
    original = write_pdf(tmp_path / "Perceptron 1958.pdf", ["Body text."])
    [result] = ingestor.ingest_files([original])
    assert (result.doc_id, result.status) == ("file:perceptron-1958", "ok")

    stored = db.get_document(ingestor.conn, "file:perceptron-1958")
    assert (stored.source, stored.title, stored.authors) == ("file", "The Perceptron", ["Frank Rosenblatt"])
    copy = Path(stored.source_uri)
    assert copy.read_bytes() == original.read_bytes()
    assert copy.parent == tmp_path / "data" / "raw" / "files"
    assert stored.metadata["original_path"] == str(original.resolve())


def test_local_file_is_skipped_until_it_changes(ingestor, tmp_path):
    original = write_pdf(tmp_path / "notes.pdf", ["Version one."])
    ingestor.ingest_files([original])
    [again] = ingestor.ingest_files([original])
    assert again.status == "skipped" and ingestor.primary.calls == 1

    write_pdf(original, ["Version two, longer."])
    [changed] = ingestor.ingest_files([original])
    assert changed.status == "ok" and ingestor.primary.calls == 2


def test_two_files_with_the_same_name_are_refused(ingestor, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ingestor.ingest_files([write_pdf(tmp_path / "a" / "notes.pdf", ["A."])])
    with pytest.raises(ValueError, match="already used by"):
        ingestor.ingest_files([write_pdf(tmp_path / "b" / "notes.pdf", ["B."])])


def test_documents_land_in_the_chosen_collection_and_stay_there(ingestor, tmp_path):
    original = write_pdf(tmp_path / "notes.pdf", ["Body text."])
    ingestor.collection = "papers"
    ingestor.ingest_files([original])
    assert db.get_document(ingestor.conn, "file:notes").collection_id == "papers"

    ingestor.collection = "hr-demo"  # same document, another collection: refused, not moved
    with pytest.raises(ValueError, match="already in collection 'papers'"):
        ingestor.ingest_files([original])
    assert db.get_document(ingestor.conn, "file:notes").collection_id == "papers"


def test_only_pdfs_are_accepted(ingestor, tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("plain text")
    with pytest.raises(ValueError, match="only PDF"):
        ingestor.ingest_files([text_file])


def test_reapply_tei_uses_saved_grobid_output(ingestor):
    tei = (Path(__file__).parent / "fixtures" / "sample.tei.xml").read_text()
    ingestor.parsed_dir.mkdir(parents=True)
    (ingestor.parsed_dir / "arxiv_1412.6980.tei.xml").write_text(tei)
    db.upsert_document(ingestor.conn, replace(make_document(), parser="grobid-0.8.2", parse_status="ok"))
    db.upsert_document(
        ingestor.conn,
        Document(
            doc_id="arxiv:2006.11239",
            source="arxiv",
            title="DDPM",
            parser="grobid-0.8.2",
            parse_status="ok",
        ),
    )

    by_id = {r.doc_id: r for r in ingestor.reapply_tei()}
    assert by_id[DOC].status == "suspect"  # the fixture is short; only the word-count gate fires
    sections = db.get_sections(ingestor.conn, DOC)
    assert sections[0].path == "Abstract" and len(sections) == 13
    assert db.get_document(ingestor.conn, DOC).title == "Adam"  # source metadata still wins
    assert by_id["arxiv:2006.11239"].status == "skipped"
    assert "no saved TEI" in by_id["arxiv:2006.11239"].issues
    assert ingestor.primary.calls == 0  # GROBID was never asked
