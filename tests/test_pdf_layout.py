"""The layout PDF parser (rag/parse/pdf.py) on small PDFs made here, one rule at a time."""

from pathlib import Path

import pymupdf
import pytest

from rag.parse.base import CAPTIONS_PATH, ParseError
from rag.parse.pdf import LayoutPDFParser, join_lines

BODY = (
    "This paragraph is ordinary body text that runs long enough to look like a real paragraph in a "
    "document. It has several sentences. It ends with a full stop."
)


def make_pdf(path: Path, pages: list[list[tuple]], toc: list | None = None) -> Path:
    """pages: per page, a list of (text, size, bold) or (text, size, bold, y) items written top down."""
    document = pymupdf.open()
    for items in pages:
        page = document.new_page()
        y = 110
        for item in items:
            text, size, bold = item[:3]
            if len(item) == 4:
                y = item[3]
            box = pymupdf.Rect(72, y, 540, y + 400)
            page.insert_textbox(box, text, fontsize=size, fontname="hebo" if bold else "helv")
            lines = max(1, len(text) // 80 + 1)
            y += lines * size * 1.3 + 14
    if toc:
        document.set_toc(toc)
    document.save(path)
    return path


def parse(path: Path):
    return LayoutPDFParser().parse_with_report(path, "file:test")


def paths(parsed) -> list[str]:
    return [s.path for s in parsed.sections]


def test_headings_from_font_size_and_boldness(tmp_path):
    pdf = make_pdf(
        tmp_path / "a.pdf",
        [
            [("Quarterly Operations Report", 22, True), ("Overview", 14, True), (BODY, 10, False)],
            [("Staffing", 14, True), (BODY, 10, False), ("Night shifts", 10, True), (BODY, 10, False)],
        ],
    )
    parsed, report = parse(pdf)
    assert report.title_from_layout == "Quarterly Operations Report"
    assert paths(parsed) == ["Overview", "Staffing", "Staffing > Night shifts"]
    assert report.headings_from.startswith("layout")
    assert parsed.sections[1].paragraphs[0].page_start == 2


def test_numbered_headings_set_the_level(tmp_path):
    pdf = make_pdf(
        tmp_path / "n.pdf",
        [
            [
                ("1 Scope", 10, False),
                (BODY, 10, False),
                ("1.1 Who this applies to", 10, False),
                (BODY, 10, False),
                ("2 Procedure", 10, False),
                (BODY, 10, False),
            ]
        ],
    )
    parsed, _ = parse(pdf)
    assert paths(parsed) == ["1 Scope", "1 Scope > 1.1 Who this applies to", "2 Procedure"]


def test_numbered_list_items_are_not_headings(tmp_path):
    items = [
        ("Steps", 14, True),
        ("1. Submit the form to your manager", 10, False),
        ("2. Wait for approval by email", 10, False),
        (BODY, 10, False),
        ("Notes", 14, True),
        (BODY, 10, False),
    ]
    parsed, _ = parse(make_pdf(tmp_path / "l.pdf", [items]))
    assert paths(parsed) == ["Steps", "Notes"]
    assert "1. Submit the form" in parsed.sections[0].text


def test_bookmarks_win_and_a_figure_label_does_not_steal_a_heading(tmp_path):
    pdf = make_pdf(
        tmp_path / "b.pdf",
        [
            [
                ("Multi-Head Attention", 8, False),
                (BODY, 10, False),
                ("2.1 Multi-Head Attention", 10, True),
                (BODY, 10, False),
                ("2.2 Training", 10, True),
                (BODY, 10, False),
                ("3 Results", 10, True),
                (BODY, 10, False),
            ],
        ],
        toc=[[1, "Multi-Head Attention", 1], [1, "Training", 1], [1, "Results", 1]],
    )
    parsed, report = parse(pdf)
    assert report.headings_from.startswith("bookmarks (3 of 3")
    assert "2.1 Multi-Head Attention" in paths(parsed)
    assert parsed.sections[0].path == "Preamble"  # the figure label stayed body text


def test_running_headers_footers_and_page_numbers_are_dropped(tmp_path):
    pages = []
    for n in range(1, 5):
        pages.append(
            [
                ("ACME Corp - Internal Use Only", 8, False, 20),
                ("Section " + str(n), 14, True, 110),
                (BODY, 10, False),
                (str(n), 9, False, 800),
            ]
        )
    parsed, report = parse(make_pdf(tmp_path / "h.pdf", pages))
    text = " ".join(s.text for s in parsed.sections)
    assert "Internal Use Only" not in text and report.dropped_header_footer == 4
    assert report.dropped_page_numbers == 4


def test_references_are_dropped_and_later_sections_kept(tmp_path):
    pdf = make_pdf(
        tmp_path / "r.pdf",
        [
            [
                ("Introduction", 14, True),
                (BODY, 10, False),
                ("References", 14, True),
                ("[1] A. Author. Some paper. 2020.", 10, False),
                ("Appendix A", 14, True),
                (BODY, 10, False),
            ]
        ],
    )
    parsed, _ = parse(pdf)
    assert paths(parsed) == ["Introduction", "Appendix A"]
    assert "Some paper" not in " ".join(s.text for s in parsed.sections)


def test_captions_go_to_their_own_section(tmp_path):
    pdf = make_pdf(
        tmp_path / "c.pdf",
        [
            [
                ("Results", 14, True),
                (BODY, 10, False),
                ("Figure 2: Accuracy by month.", 9, False),
                ("Method", 14, True),
                (BODY, 10, False),
            ]
        ],
    )
    parsed, _ = parse(pdf)
    assert paths(parsed) == ["Results", "Method", CAPTIONS_PATH]
    assert parsed.sections[-1].text == "Figure 2: Accuracy by month."


def test_a_paragraph_cut_by_a_page_break_is_joined(tmp_path):
    pdf = make_pdf(
        tmp_path / "p.pdf",
        [
            [
                ("Policy", 14, True),
                (BODY, 10, False),
                ("Employees who travel must book through the portal and", 10, False),
            ],
            [("keep every receipt for thirty days.", 10, False), ("Other", 14, True), (BODY, 10, False)],
        ],
    )
    parsed, report = parse(pdf)
    joined = [p for p in parsed.sections[0].paragraphs if "portal and keep every receipt" in parsed.sections[0].text]
    assert report.joined_across_pages == 1 and joined
    last = parsed.sections[0].paragraphs[-1]
    assert (last.page_start, last.page_end) == (1, 2)


def test_no_headings_falls_back_to_pages(tmp_path):
    parsed, _ = parse(make_pdf(tmp_path / "f.pdf", [[(BODY, 10, False)], [(BODY, 10, False)]]))
    assert paths(parsed) == ["Page 1", "Page 2"]
    assert "no headings found: one section per page" in parsed.issues


def test_a_pdf_without_text_needs_ocr(tmp_path):
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    document.save(tmp_path / "scan.pdf")
    with pytest.raises(ParseError, match="needs OCR"):
        LayoutPDFParser().parse(tmp_path / "scan.pdf", "file:scan")


def test_join_lines():
    assert join_lines(["inter-", "national trade"]) == "international trade"
    assert join_lines(["our 152-", "layer net"]) == "our 152-layer net"
    assert join_lines(["COVID-", "19 cases"]) == "COVID-19 cases"
    assert join_lines(["we deﬁne the", "ﬂow"]) == "we define the flow"
