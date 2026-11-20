"""Last-resort parser: plain text blocks from PyMuPDF, one section per page.

Used only when GROBID fails on a PDF. It knows nothing about headings,
columns or captions, so its output is always marked "suspect". What it does
keep is exact page numbers, so citations still point at the right page.
Text after a "References" heading is dropped.

PyMuPDF is AGPL-licensed: fine for an open-source repo; swap in Docling (MIT)
if this code ever ships closed-source.
"""

import re
from pathlib import Path

import pymupdf

from rag.models import Section
from rag.parse.base import ParsedDocument, ParseError, Parser, quality_issues

REFERENCES_HEADING = re.compile(r"^(references|bibliography)$", re.IGNORECASE)
FALLBACK_ISSUE = "fallback parser: no section structure"
TEXT_BLOCK = 0


class PyMuPDFParser(Parser):
    name = "pymupdf"

    def parse(self, pdf_path: Path, doc_id: str) -> ParsedDocument:
        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:  # PyMuPDF raises several unrelated types for bad files
            raise ParseError(f"PyMuPDF could not open {pdf_path.name}: {exc}") from exc

        sections: list[Section] = []
        with document:
            n_pages = document.page_count
            title = " ".join((document.metadata or {}).get("title", "").split())
            for page in document:
                page_number = page.number + 1
                paragraphs = []
                reached_references = False
                for block in page.get_text("blocks"):
                    if block[6] != TEXT_BLOCK:
                        continue
                    text = " ".join(block[4].split())
                    if REFERENCES_HEADING.match(text):
                        reached_references = True
                        break
                    if text:
                        paragraphs.append((text, page_number, page_number))
                if paragraphs:
                    path = f"Page {page_number}"
                    sections.append(Section.from_paragraphs(doc_id, len(sections), path, 1, path, paragraphs))
                if reached_references:
                    break

        if not sections:
            raise ParseError(f"PyMuPDF found no text in {pdf_path.name} (scanned PDF?)")
        parsed = ParsedDocument(
            doc_id=doc_id,
            parser=f"pymupdf-{pymupdf.VersionBind}",
            title=title,
            authors=[],
            abstract="",
            sections=sections,
            n_pages=n_pages,
            n_references=0,
        )
        parsed.issues = [FALLBACK_ISSUE, *quality_issues(parsed)]
        return parsed


HORIZONTAL = (1, 0)
FONT_SIZE_TOLERANCE = 0.5
MAX_FONT_SIZES_TRIED = 3


def pdf_title_candidates(pdf_path: Path) -> dict[str, str]:
    """Two title guesses straight from the PDF: its metadata title, and the largest horizontal text
    on page 1 (rotated text such as arXiv's side stamp is skipped). The largest size whose text
    passes the title checks wins; title lines that wrap are joined."""
    from rag.parse.title import title_problem

    candidates = {"pdf metadata": "", "largest font": ""}
    try:
        document = pymupdf.open(pdf_path)
    except Exception:  # an unreadable PDF just has no candidates; the parse reports the real error
        return candidates
    with document:
        candidates["pdf metadata"] = (document.metadata or {}).get("title", "") or ""
        if document.page_count == 0:
            return candidates
        lines: list[tuple[float, float, str]] = []  # (font size, top y, text)
        for block in document[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if tuple(round(v) for v in line["dir"]) != HORIZONTAL:
                    continue
                # Spans of one line are joined without spaces: small caps come as "A" + "DAM".
                text = "".join(span["text"] for span in line["spans"]).strip()
                if text:
                    lines.append((max(span["size"] for span in line["spans"]), line["bbox"][1], text))
    sizes = sorted({round(size, 1) for size, _, _ in lines}, reverse=True)
    for size in sizes[:MAX_FONT_SIZES_TRIED]:
        text = " ".join(t for s, _, t in sorted(lines, key=lambda x: x[1]) if abs(s - size) <= FONT_SIZE_TOLERANCE)
        if title_problem(text) is None:
            candidates["largest font"] = text
            break
    return candidates
