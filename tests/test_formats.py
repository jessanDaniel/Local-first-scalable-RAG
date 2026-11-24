"""Step 21b part B: Word and HTML files, and folders walked recursively."""

import zipfile
from pathlib import Path

import pytest

from rag.config import Settings
from rag.ingest import Ingestor, find_documents
from rag.parse.base import ParseError
from rag.parse.office import DocxParser, HTMLDocParser
from rag.store import db

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
LONG = " ".join(["Employees must submit expense reports within thirty days of travel."] * 16)


def para(text: str, style: str = "", numbered: bool = False) -> str:
    props = ""
    if style or numbered:
        props = "<w:pPr>" + (f'<w:pStyle w:val="{style}"/>' if style else "")
        props += '<w:numPr><w:ilvl w:val="0"/></w:numPr>' if numbered else ""
        props += "</w:pPr>"
    return f"<w:p>{props}<w:r><w:t>{text}</w:t></w:r></w:p>"


def make_docx(path: Path, body: str, title: str = "") -> Path:
    styles = (
        f"<w:styles {W}>"
        '<w:style w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
        '<w:style w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>'
        '<w:style w:styleId="Title"><w:name w:val="Title"/></w:style>'
        "</w:styles>"
    )
    core = (
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></cp:coreProperties>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f"<w:document {W}><w:body>{body}</w:body></w:document>")
        archive.writestr("word/styles.xml", styles)
        archive.writestr("docProps/core.xml", core)
    return path


# --- Word ---------------------------------------------------------------------------------------


def test_docx_headings_come_from_paragraph_styles(tmp_path):
    body = (
        para("Travel Policy", "Title")
        + para("Booking", "Heading1")
        + para(LONG)
        + para("Flights", "Heading2")
        + para("Book economy for flights under six hours.")
        + para("Use the portal", numbered=True)
        + "<w:tbl><w:tr><w:tc>"
        + para("Class")
        + "</w:tc><w:tc>"
        + para("Limit")
        + "</w:tc></w:tr>"
        + "<w:tr><w:tc>"
        + para("Economy")
        + "</w:tc><w:tc>"
        + para("Any")
        + "</w:tc></w:tr></w:tbl>"
        + para("Hotels", "Heading1")
        + para(LONG)
    )
    parsed = DocxParser().parse(make_docx(tmp_path / "t.docx", body), "file:t")
    assert parsed.title == "Travel Policy"
    assert [s.path for s in parsed.sections] == ["Booking", "Booking > Flights", "Hotels"]
    flights = parsed.sections[1].text
    assert "- Use the portal" in flights and "Class | Limit" in flights and "Economy | Any" in flights
    assert parsed.sections[0].paragraphs[0].page_start is None  # Word has no fixed pages


def test_docx_title_from_document_properties(tmp_path):
    parsed = DocxParser().parse(
        make_docx(tmp_path / "t.docx", para("Body", "Heading1") + para(LONG), "Handbook"), "f:t"
    )
    assert parsed.title == "Handbook"


def test_not_a_word_file_is_a_parse_error(tmp_path):
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ParseError, match="cannot read"):
        DocxParser().parse(bad, "file:bad")


# --- HTML ---------------------------------------------------------------------------------------


def test_html_keeps_content_and_drops_page_furniture(tmp_path):
    page = tmp_path / "p.html"
    page.write_text(
        "<html><head><title>Leave Policy</title><style>p{color:red}</style></head><body>"
        "<nav><a href='/'>Home</a> Menu</nav><header>Site banner</header>"
        "<h1>Leave Policy</h1><p>Intro text about leave.</p>"
        "<h2>Annual leave</h2><p>Employees get <b>25 days</b> &amp; public holidays.</p>"
        "<ul><li>Book ahead</li><li>Tell your manager</li></ul>"
        "<table><tr><th>Years</th><th>Days</th></tr><tr><td>1-3</td><td>25</td></tr></table>"
        "<h2>Sick leave</h2><div>Call in before 9am.</div>"
        "<script>alert('x')</script><footer>Copyright</footer></body></html>"
    )
    parsed = HTMLDocParser().parse(page, "file:p")
    assert parsed.title == "Leave Policy"
    # One <h1> names the page: it is the title, and h2 becomes the top level.
    assert [s.path for s in parsed.sections] == ["Preamble", "Annual leave", "Sick leave"]
    text = " ".join(s.text for s in parsed.sections)
    assert "25 days & public holidays" in text and "- Book ahead" in text and "1-3 | 25" in text
    for furniture in ("Menu", "Site banner", "alert", "Copyright", "color:red"):
        assert furniture not in text


# --- folders ------------------------------------------------------------------------------------


def test_folders_are_walked_recursively_with_a_skip_report(tmp_path):
    root = tmp_path / "docs"
    (root / "policies" / "old").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "a.md").write_text("# A\n\ntext a")
    (root / "policies" / "travel.txt").write_text("travel")
    (root / "policies" / "old" / "travel.txt").write_text("old travel")  # same name, other folder
    (root / "policies" / "copy.txt").write_text("travel")  # same content as travel.txt
    (root / "policies" / "empty.md").write_text("")
    (root / ".git" / "config.txt").write_text("hidden")
    (root / ".DS_Store").write_text("x")
    (root / "~$draft.docx").write_text("lock")
    (root / "sheet.xlsx").write_text("x")
    found, skipped = find_documents([root])
    assert [(f.path.relative_to(root).as_posix(), f.key) for f in found] == [
        ("a.md", "a"),
        ("policies/copy.txt", "policies-copy"),
        ("policies/old/travel.txt", "policies-old-travel"),
    ]
    reasons = {p.relative_to(root).as_posix(): why for p, why in skipped}
    assert reasons["policies/travel.txt"].startswith("same content as")
    assert reasons["policies/empty.md"] == "empty"
    assert reasons[".git"] == "hidden folder" and reasons[".DS_Store"] == "hidden or temporary file"
    assert reasons["~$draft.docx"] == "hidden or temporary file"
    assert reasons["sheet.xlsx"] == "unsupported type .xlsx"


def test_a_folder_of_mixed_formats_ingests_without_grobid(tmp_path):
    root = tmp_path / "docs"
    (root / "hr").mkdir(parents=True)
    make_docx(
        root / "hr" / "Travel.docx", para("Booking", "Heading1") + para(LONG) + para("Hotels", "Heading1") + para(LONG)
    )
    (root / "hr" / "leave.htm").write_text(f"<h1>Leave</h1><h2>Annual</h2><p>{LONG}</p><h2>Sick</h2><p>{LONG}</p>")
    (root / "readme.md").write_text(f"# Readme\n\n## Setup\n\n{LONG}\n\n## Use\n\n{LONG}\n")
    ing = Ingestor(
        Settings(_env_file=None, data_dir=tmp_path / "data", grobid_url="http://127.0.0.1:9"), collection="docs"
    )
    db.ensure_collection(ing.conn, "docs")
    found, _ = find_documents([root])
    results = ing.ingest_files(found)
    stored = {d.doc_id: d for d in db.list_documents(ing.conn)}
    ing.close()
    by_id = {r.doc_id: r for r in results}
    assert by_id["file:readme"].status == by_id["file:hr-leave"].status == "ok"
    # No title inside the Word file: the original file name, flagged as a guess
    assert (by_id["file:hr-travel"].title, by_id["file:hr-travel"].issues) == ("Travel", "title guessed from file name")
    assert set(stored) == {"file:hr-travel", "file:hr-leave", "file:readme"}
    assert stored["file:hr-leave"].source_format == "html"
    assert Path(stored["file:hr-leave"].source_uri).name == "file_hr-leave.html"  # .htm stored as .html
    assert stored["file:hr-travel"].source_format == "docx"
