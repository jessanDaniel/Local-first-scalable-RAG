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
                    sections.append(
                        Section.from_paragraphs(doc_id, len(sections), path, 1, path, paragraphs)
                    )
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
