from pathlib import Path

import pytest

from rag.parse.base import ParseError
from rag.parse.text import PREAMBLE, TextParser, looks_like_heading

DOCUMENTS = Path(__file__).parents[1] / "documents"


def parse(tmp_path, text: str, name: str = "doc.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return TextParser().parse(path, "file:doc")


@pytest.mark.parametrize(
    ("line", "heading"),
    [
        ("Section 2: Working Hours and Attendance", True),
        ("Appendix A: Quick Reference Contacts", True),
        ("3.1 Results", True),
        ("Refunds are processed within 5-7 business days.", False),  # sentence
        ("Notes:", False),  # ends in punctuation
        ("lowercase start is not a heading", False),
        ("A very long line that goes on and on with many words well beyond any heading", False),
    ],
)
def test_heading_rule(line, heading):
    assert looks_like_heading(line) is heading


def test_plain_text_sections_title_and_preamble(tmp_path):
    text = (
        "Field Guide\nVersion: 2\n\nIntro paragraph that is\nhard wrapped across lines.\n\n"
        "Section 1: Birds\n\nBirds fly.\n\nThey also sing.\n\nSection 2: Fish\n\nFish swim.\n"
    )
    parsed = parse(tmp_path, text)
    assert parsed.title == "Field Guide"
    assert [s.path for s in parsed.sections] == [PREAMBLE, "Section 1: Birds", "Section 2: Fish"]
    assert parsed.sections[0].text.split("\n\n") == ["Version: 2", "Intro paragraph that is hard wrapped across lines."]
    assert len(parsed.sections[1].paragraphs) == 2
    assert all(p.page_start is None for s in parsed.sections for p in s.paragraphs)


def test_a_heading_like_last_line_is_kept_as_text(tmp_path):
    parsed = parse(tmp_path, "Title Here\n\nSection 1: Only\n\nBody text.\n\nSigned The Team\n")
    assert parsed.sections[-1].text.endswith("Signed The Team")


def test_markdown_tree_title_lists_and_code(tmp_path):
    text = (
        "# Runbook\n\nWhy this exists.\n\n## Deploy\nSteps:\n\n- build\n- ship\n\n"
        "### Rollback\n\n```bash\n# not a heading\ngit revert HEAD\n```\n\n## Monitor\n\nWatch the graphs.\n"
    )
    parsed = parse(tmp_path, text, "runbook.md")
    assert parsed.title == "Runbook"
    assert [s.path for s in parsed.sections] == [PREAMBLE, "Deploy", "Deploy > Rollback", "Monitor"]
    assert "- build\n- ship" in parsed.sections[1].text
    assert "# not a heading\ngit revert HEAD" in parsed.sections[2].text
    assert [s.level for s in parsed.sections] == [1, 1, 2, 1]


def test_markdown_with_several_top_headings_has_no_title_taken_from_them(tmp_path):
    parsed = parse(tmp_path, "# One\n\nA.\n\n# Two\n\nB.\n", "x.md")
    assert parsed.title == ""
    assert [s.path for s in parsed.sections] == ["One", "Two"]


@pytest.mark.parametrize("content", ["", "   \n\n"])
def test_empty_files_are_parse_errors(tmp_path, content):
    with pytest.raises(ParseError, match="empty"):
        parse(tmp_path, content)


def test_windows_line_endings(tmp_path):
    parsed = parse(tmp_path, "Title\r\n\r\nSection 1: A\r\n\r\nText.\r\n")
    assert parsed.sections[0].path == "Section 1: A"


@pytest.mark.parametrize(
    ("name", "title", "n_sections", "first", "last"),
    [
        (
            "company_policy",
            "Acme Corp Customer Policy Handbook",
            8,
            "Section 1: Refund Policy",
            "Section 8: Order Cancellation",
        ),
        ("employee_handbook", "Acme Corp Employee Handbook", 21, PREAMBLE, "Appendix B: Related Documents"),
    ],
)
def test_real_hr_documents(name, title, n_sections, first, last):
    parsed = TextParser().parse(DOCUMENTS / f"{name}.txt", f"file:{name}")
    assert (parsed.title, len(parsed.sections), parsed.status) == (title, n_sections, "ok")
    assert (parsed.sections[0].path, parsed.sections[-1].path) == (first, last)
