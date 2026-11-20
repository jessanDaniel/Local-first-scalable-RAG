import pymupdf
import pytest

from rag.parse.pymupdf import pdf_title_candidates
from rag.parse.title import choose_title, clean_title, title_problem


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Attention Is All You Need", None),
        ("Adam: A Method for Stochastic Optimization", None),
        ("Eﬃcient Estimation of Word Representations", None),  # ligature is fine once cleaned
        ("Permission-Based Access Control for Licensed Software", None),  # notice words, not a notice
        ("Copyright 2021 by the authors", "licence or copyright notice"),
        # What GROBID returned in step 9:
        (
            "Provided proper attribution is provided, Google hereby grants permission to reproduce the tables",
            "licence or copyright notice",
        ),
        ("Published as a conference paper at ICLR 2015 ADAM: A METHOD FOR STOCHASTIC", "venue banner"),
        ("arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", "arXiv stamp"),
        ("Under review as a conference paper at ICLR 2024", "review or preprint banner"),
        ("Microsoft Word - draft3.docx", "authoring-tool placeholder"),
        ("Contact: someone@example.com", "contains an email or URL"),
        ("", "empty"),
        ("12345 67890", "mostly not letters"),
        (" ".join(["word"] * 40), "too long"),
    ],
)
def test_title_problem(text, problem):
    assert title_problem(text) == problem


def test_clean_title_expands_ligatures_and_spaces():
    assert clean_title("  Eﬃcient\n Estimation ") == "Efficient Estimation"


def test_first_usable_candidate_wins_and_rejections_are_recorded():
    title, source, rejected = choose_title(
        [("parser", "Published as a conference paper at ICLR"), ("pdf metadata", ""), ("largest font", "Real Title")]
    )
    assert (title, source) == ("Real Title", "largest font")
    assert rejected == ["parser title rejected (venue banner): Published as a conference paper at ICLR"]


def test_the_last_candidate_is_used_even_if_it_fails():
    assert choose_title([("parser", ""), ("file name", "notes")])[:2] == ("notes", "file name")


def test_pdf_candidates_skip_rotated_stamps_and_unusable_sizes(tmp_path):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((20, 400), "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", fontsize=24, rotate=90)  # side stamp
    page.insert_text((72, 60), "Provided proper attribution is provided, Google hereby grants", fontsize=20)
    page.insert_text((72, 120), "Attention Is All", fontsize=17)
    page.insert_text((72, 142), "You Need", fontsize=17)  # a title that wraps onto two lines
    page.insert_text((72, 200), "Body text in a small font.", fontsize=10)
    document.set_metadata({"title": "Microsoft Word - paper.docx"})
    path = tmp_path / "paper.pdf"
    document.save(path)

    candidates = pdf_title_candidates(path)
    assert candidates["largest font"] == "Attention Is All You Need"
    assert candidates["pdf metadata"] == "Microsoft Word - paper.docx"  # raw; choose_title rejects it
    title, source, _ = choose_title(
        [("parser", "Provided proper attribution is provided"), *candidates.items(), ("file name", "paper")]
    )
    assert (title, source) == ("Attention Is All You Need", "largest font")


def test_unreadable_pdf_has_no_candidates(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    assert pdf_title_candidates(path) == {"pdf metadata": "", "largest font": ""}
