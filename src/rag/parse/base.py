"""The parser contract: a file in, a ParsedDocument out.

Every parser (GROBID for scholarly PDFs, PyMuPDF as the fallback, later
others per format) returns the same ParsedDocument, so the rest of the
pipeline never knows which one ran. Metadata the source knows better than the
file (arXiv title, authors, dates) comes from the source during ingest; the
parser's title/authors are used only when the source has none (local files).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from rag.models import ParseStatus, Section

ABSTRACT_PATH = "Abstract"
CAPTIONS_PATH = "Figures and Tables"
APPENDIX_PATH = "Appendix"

# Quality gates. A parse that trips any of them is kept but marked "suspect".
MIN_BODY_WORDS = 300
MIN_BODY_SECTIONS = 2
MIN_LETTER_RATIO = 0.5  # letters / non-space characters; broken font maps fall far below


class ParseError(RuntimeError):
    """The parser could not produce any output for this PDF."""


class ParserUnavailableError(ParseError):
    """The parser's service is down. Not this PDF's fault: stop instead of falling back."""


@dataclass
class ParsedDocument:
    doc_id: str
    parser: str  # e.g. "grobid-0.8.2"
    title: str
    authors: list[str]
    abstract: str
    sections: list[Section]
    n_pages: int | None
    n_references: int
    issues: list[str] = field(default_factory=list)

    @property
    def status(self) -> ParseStatus:
        return "suspect" if self.issues else "ok"

    @property
    def body_sections(self) -> list[Section]:
        return [s for s in self.sections if s.path not in (ABSTRACT_PATH, CAPTIONS_PATH)]


class Parser(ABC):
    name: str

    @abstractmethod
    def parse(self, pdf_path: Path, doc_id: str) -> ParsedDocument:
        raise NotImplementedError


def quality_issues(document: ParsedDocument) -> list[str]:
    issues = []
    if not document.title:
        issues.append("no title")
    body = document.body_sections
    if len(body) < MIN_BODY_SECTIONS:
        issues.append(f"only {len(body)} body sections")
    body_text = " ".join(s.text for s in body)
    n_words = len(body_text.split())
    if n_words < MIN_BODY_WORDS:
        issues.append(f"only {n_words} body words")
    chars = [c for c in body_text if not c.isspace()]
    if chars:
        letter_ratio = sum(c.isalpha() for c in chars) / len(chars)
        if letter_ratio < MIN_LETTER_RATIO:
            issues.append(f"letter ratio {letter_ratio:.2f} (garbled text?)")
    return issues


def print_parse_debug(document: ParsedDocument) -> None:
    print(f"Document:   {document.doc_id}")
    print(f"Parser:     {document.parser}")
    print(f"Title:      {document.title}")
    print(f"Authors:    {', '.join(document.authors)}")
    print(f"Pages:      {document.n_pages}   References: {document.n_references}")
    print(f"Abstract:   {len(document.abstract.split())} words")
    print(f"Status:     {document.status}" + (f"  ({'; '.join(document.issues)})" if document.issues else ""))
    body_words = sum(len(s.text.split()) for s in document.body_sections)
    print(f"Sections:   {len(document.sections)} ({len(document.body_sections)} body, {body_words} body words)")
    for s in document.sections:
        pages = sorted({p.page_start for p in s.paragraphs if p.page_start is not None}
                       | {p.page_end for p in s.paragraphs if p.page_end is not None})
        page_range = f"p.{pages[0]}-{pages[-1]}" if pages else "p.?"
        print(f"  {s.ord:>3} L{s.level} {page_range:>8} paras={len(s.paragraphs):>3} "
              f"words={len(s.text.split()):>5}  {s.path[:80]}")
