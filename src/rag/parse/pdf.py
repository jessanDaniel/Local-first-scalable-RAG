"""The default PDF parser: sections, paragraphs and pages from PyMuPDF's text layout. No service.

GROBID knows scientific papers; this parser assumes nothing about the kind of document. Steps,
each one visible in `rag parse --parser pymupdf <pdf>`:

 1. Lines from every page, with font size, boldness and position. Rotated text (arXiv's side
    stamp) is skipped. A PDF with almost no text has no text layer: ParseError("needs OCR").
 2. Running headers and footers go: a line near the top or bottom of the page whose text (digits
    ignored) repeats on at least 40% of pages, and bare page numbers ("12", "Page 3 of 10").
 3. Body size = the font size holding most characters.
 4. Headings, from the PDF's bookmarks (its table of contents) when it has at least 3 entries that
    can be found on their pages; otherwise from the text itself: a short block (at most 15 words,
    not ending like a sentence) that is larger than the body, or bold throughout. Levels come from
    numbering ("2.3" = level 2) or else from font size.
 5. The largest text on page 1, if it occurs nowhere else, is the title rather than a section.
 6. Paragraphs = PyMuPDF's text blocks, lines joined, words hyphenated across lines rejoined. A
    paragraph cut by a page break (no closing punctuation, next page starts in lower case) is
    joined, keeping both pages. "Figure 3:" / "Table 2." blocks go to a captions section.
 7. A "References" / "Bibliography" section is dropped (its text is citations, not content);
    sections after it (appendices) are kept.

If fewer than 2 headings are found, sections fall back to one per page, flagged "no headings".
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from rag.models import Section
from rag.parse.base import CAPTIONS_PATH, ParsedDocument, ParseError, Parser, quality_issues

VERSION = "pdf-layout-v1"
BOLD = 16  # PyMuPDF span flag
HORIZONTAL = (1, 0)
MARGIN = 0.08  # top / bottom share of the page where running headers and footers live
REPEAT_SHARE = 0.4
MIN_CHARS_PER_PAGE = 10  # below this on average: no text layer (a scanned PDF has ~0)
SIZE_TOLERANCE = 0.4
MAX_HEADING_WORDS = 15
MAX_HEADING_CHARS = 140
NUMBERED = re.compile(
    r"^(?:(?:section|chapter|part)\s+)?(\d+(?:\.\d+){0,4}|[A-Z](?:\.\d+){0,3}|[IVX]+)[.)]?\s+\S", re.I
)
PAGE_NUMBER = re.compile(r"^(?:page\s+)?[-–]?\s*\d{1,4}\s*[-–]?(?:\s+of\s+\d{1,4})?$|^[ivxlc]{1,6}$", re.I)
CAPTION = re.compile(r"^(?:figure|fig\.|table)\s+[A-Z]?\d+[.:]", re.I)
REFERENCES = re.compile(r"^(?:\d+\.?\s+)?(references|bibliography|works cited|literature cited)$", re.I)
SENTENCE_END = (".", ",", ";", ":")


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float


@dataclass
class Block:
    page: int  # 1-based
    lines: list[Line]
    page_height: float
    heading_level: int | None = None
    drop: bool = False

    @property
    def text(self) -> str:
        return join_lines([line.text for line in self.lines])

    @property
    def size(self) -> float:
        return max(line.size for line in self.lines)

    @property
    def bold(self) -> bool:
        return all(line.bold for line in self.lines)


@dataclass
class LayoutReport:
    """What the parser decided, for the debug view and the hand check."""

    headings_from: str = ""
    body_size: float = 0.0
    dropped_header_footer: int = 0
    dropped_page_numbers: int = 0
    title_from_layout: str = ""
    joined_across_pages: int = 0
    notes: list[str] = field(default_factory=list)


LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})


def join_lines(lines: list[str]) -> str:
    """Lines of one block -> one paragraph. "inter-" + "national" -> "international"; a hyphen
    after a digit or before an upper-case letter stays, without a space ("152-" + "layer" ->
    "152-layer", "COVID-" + "19"). Ligature characters ("ﬁ") become letters."""
    out = ""
    for raw in lines:
        line = " ".join(raw.translate(LIGATURES).split())
        if not line:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalnum():
            out = out + line
        else:
            out = f"{out} {line}" if out else line
    return out


def join_spans(spans: list[dict]) -> str:
    """Spans of one line -> its text. Small caps come as separate spans ("I" + "NITIALIZATION" +
    "B" + "IAS") with no space characters: a visible gap between two spans is a space."""
    text = ""
    previous = None
    for span in spans:
        if previous is not None:
            gap = span["bbox"][0] - previous["bbox"][2]
            spaced = text.endswith(" ") or span["text"].startswith(" ")
            if gap > 0.2 * min(span["size"], previous["size"]) and not spaced:
                text += " "
        text += span["text"]
        previous = span
    return text


def normalized(text: str) -> str:
    """For comparing headers, footers and bookmark titles: digits ignored, case and spaces folded."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip().lower()


def read_blocks(document) -> list[Block]:
    blocks = []
    for page in document:
        height = page.rect.height or 1.0
        for raw in page.get_text("dict")["blocks"]:
            lines = []
            for line in raw.get("lines", []):
                if tuple(round(v) for v in line["dir"]) != HORIZONTAL:
                    continue
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = join_spans(spans)
                chars = sum(len(s["text"].strip()) for s in spans)
                bold_chars = sum(len(s["text"].strip()) for s in spans if s["flags"] & BOLD)
                size = max(s["size"] for s in spans)
                lines.append(Line(text, size, bold_chars >= 0.8 * chars, line["bbox"][1], line["bbox"][3]))
            if lines:
                blocks.append(Block(page.number + 1, lines, height))
    return blocks


def drop_headers_and_footers(blocks: list[Block], n_pages: int, report: LayoutReport) -> None:
    def in_margin(block: Block) -> bool:
        top = block.lines[0].y0 / block.page_height
        bottom = block.lines[-1].y1 / block.page_height
        return top < MARGIN or bottom > 1 - MARGIN

    pages_with: dict[str, set[int]] = {}
    for block in blocks:
        if in_margin(block):
            pages_with.setdefault(normalized(block.text), set()).add(block.page)
    for block in blocks:
        if not in_margin(block):
            continue
        text = block.text
        if PAGE_NUMBER.match(text.strip()):
            block.drop = True
            report.dropped_page_numbers += 1
        elif n_pages >= 3 and len(pages_with[normalized(text)]) >= max(2, REPEAT_SHARE * n_pages):
            block.drop = True
            report.dropped_header_footer += 1


def body_size(blocks: list[Block]) -> float:
    chars = Counter()
    for block in blocks:
        for line in block.lines:
            chars[round(line.size * 2) / 2] += len(line.text)
    return chars.most_common(1)[0][0] if chars else 10.0


def heading_candidate(block: Block, body: float) -> bool:
    text = block.text.strip()
    if not text or len(text) > MAX_HEADING_CHARS or len(text.split()) > MAX_HEADING_WORDS:
        return False
    if text.endswith(SENTENCE_END) and not NUMBERED.match(text) or CAPTION.match(text) or PAGE_NUMBER.match(text):
        return False
    if not any(c.isalpha() for c in text):
        return False
    larger = block.size >= body + 1.0
    bold = block.bold and block.size >= body - SIZE_TOLERANCE
    return larger or bold or (len(block.lines) <= 2 and (numbered_title(text) or all_caps(text)))


def title_case_share(words: list[str]) -> float:
    words = [w for w in words if w[:1].isalpha() and len(w) > 3]  # skip "of", "and", "the"
    return sum(w[0].isupper() for w in words) / len(words) if words else 0.0


def numbered_title(text: str) -> bool:
    """ "2.1 ADAM'S UPDATE RULE", "3 Method", "4.2 Single-Network Classification" - but not "1. Submit
    the form" (a full stop after a single number is a list item)."""
    match = re.match(r"^(\d+(?:\.\d+)+|\d+)\s+(\S.*)$", text)
    if not match or len(text.split()) > 12:
        return False
    rest = match.group(2).split()
    if "." in match.group(1):  # "1.1 Who this applies to": multi-level numbers are rarely list items
        return rest[0][:1].isupper()
    return all_caps(match.group(2)) or title_case_share(rest) >= 0.6


def all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(text.split()) >= 2 and len(letters) >= 6 and all(c.isupper() for c in letters)


def numbering_depth(text: str) -> int | None:
    match = NUMBERED.match(text)
    if not match:
        return None
    token = match.group(1)
    return token.count(".") + 1 if token[0].isdigit() or "." in token else 1


def headings_from_layout(blocks: list[Block], body: float) -> int:
    candidates = [b for b in blocks if not b.drop and heading_candidate(b, body)]
    # Size ranks, largest first: the biggest heading size is level 1.
    sizes = sorted({round(b.size * 2) / 2 for b in candidates}, reverse=True)
    rank = {size: i + 1 for i, size in enumerate(sizes)}
    for block in candidates:
        depth = numbering_depth(block.text)
        block.heading_level = min(depth if depth else rank[round(block.size * 2) / 2], 4)
    return len(candidates)


STRUCTURAL = re.compile(
    r"^(?:[\dA-Z]+\.?\s+)?(abstract|summary|executive summary|references|bibliography|acknowledge?ments?"
    r"|appendix(?:\s+\S+)?|appendices|glossary|index)$",
    re.I,
)


def headings_from_bookmarks(blocks: list[Block], toc: list, body: float) -> int:
    """Mark the block each bookmark points at: on its page, text equal to the bookmark title (with or
    without its number). A block that looks like a heading wins over one that only has the words
    (a figure label "Multi-Head Attention" vs the heading "3.2.2 Multi-Head Attention"). Standard
    headings that PDFs often leave out of their bookmarks (Abstract, References, Appendix ...) are
    taken from the layout."""
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        if not block.drop:
            by_page.setdefault(block.page, []).append(block)
    found = 0
    for level, title, page in toc:
        want = normalized(title)
        if not want:
            continue
        matches = []
        for block in by_page.get(page, []):
            text = normalized(block.text)
            stripped = normalized(re.sub(r"^[\dA-Z]+(?:\.\d+)*[.)]?\s+", "", block.text))
            if block.heading_level is None and len(text) <= len(want) + 12 and want in (text, stripped):
                matches.append(block)
        if matches:
            best = next((b for b in matches if heading_candidate(b, body)), matches[0])
            best.heading_level = min(level, 4)
            found += 1
    for block in blocks:
        if block.heading_level is None and not block.drop and STRUCTURAL.match(block.text.strip()):
            if heading_candidate(block, body):
                block.heading_level = 1
    return found


def title_block(blocks: list[Block], body: float) -> Block | None:
    """The largest text on page 1, if it is clearly larger than the body, the only block that size on
    page 1, and no later page has text as large (a repeated size is a heading level, not a title)."""
    page1 = [b for b in blocks if b.page == 1 and not b.drop]
    if not page1:
        return None
    largest = max(page1, key=lambda b: b.size)
    if largest.size < body + 2.0 or len(largest.text.split()) > 30:
        return None
    if any(b is not largest and b.size >= largest.size - SIZE_TOLERANCE for b in blocks if not b.drop):
        return None
    return largest


class LayoutPDFParser(Parser):
    name = VERSION

    def parse(self, pdf_path: Path, doc_id: str) -> ParsedDocument:
        parsed, _ = self.parse_with_report(pdf_path, doc_id)
        return parsed

    def parse_with_report(self, pdf_path: Path, doc_id: str) -> tuple[ParsedDocument, LayoutReport]:
        from rag.parse.text import _SectionBuilder

        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:  # PyMuPDF raises several unrelated types for bad files
            raise ParseError(f"PyMuPDF could not open {pdf_path.name}: {exc}") from exc
        report = LayoutReport()
        with document:
            n_pages = document.page_count
            blocks = read_blocks(document)
            toc = document.get_toc(simple=True)
        n_chars = sum(len(b.text) for b in blocks)
        if n_pages and n_chars / n_pages < MIN_CHARS_PER_PAGE:
            raise ParseError(
                f"{pdf_path.name}: {n_chars} characters of text in {n_pages} pages: "
                "no text layer (scanned?) - needs OCR, which is not supported yet"
            )

        drop_headers_and_footers(blocks, n_pages, report)
        report.body_size = body = body_size([b for b in blocks if not b.drop])

        title = title_block(blocks, body)
        if title is not None:
            title.drop = True
            report.title_from_layout = title.text

        found = headings_from_bookmarks(blocks, toc, body) if len(toc) >= 3 else 0
        if found >= 3:
            report.headings_from = f"bookmarks ({found} of {len(toc)} found on their pages)"
        else:
            for block in blocks:
                block.heading_level = None
            found = headings_from_layout(blocks, body)
            report.headings_from = f"layout ({found} headings; body {body:g} pt)"

        builder = _SectionBuilder(doc_id)
        captions: list[tuple[str, int, int]] = []
        issues: list[str] = []
        if found < 2:
            issues.append("no headings found: one section per page")
            report.notes.append("fewer than 2 headings: sections are pages")
        in_references = False
        pending: tuple[str, int, int] | None = None  # a paragraph that may continue on the next page

        def flush_pending() -> None:
            nonlocal pending
            if pending and not in_references:
                builder.paragraph(*pending)
            pending = None

        current_page = 0
        for block in blocks:
            if block.drop:
                continue
            if found < 2 and block.page != current_page:
                flush_pending()
                builder.heading(f"Page {block.page}", level=1)
            current_page = block.page
            text = block.text
            if block.heading_level is not None and found >= 2:
                flush_pending()
                in_references = bool(REFERENCES.match(text.strip()))
                if not in_references:
                    builder.heading(text.strip(), block.heading_level)
                continue
            if found < 2 and REFERENCES.match(text.strip()):
                flush_pending()  # no headings: a "References" line ends the content for good
                in_references = True
                continue
            if in_references:
                continue
            if CAPTION.match(text):
                captions.append((text, block.page, block.page))
                continue
            if pending:
                prev_text, start, _ = pending
                if block.page > pending[2] and not prev_text.endswith((".", "!", "?", ":")) and text[:1].islower():
                    pending = (join_lines([prev_text, text]), start, block.page)
                    report.joined_across_pages += 1
                    continue
                flush_pending()
            pending = (text, block.page, block.page)
        flush_pending()
        sections = builder.finish()
        if captions:
            sections.append(Section.from_paragraphs(doc_id, len(sections), CAPTIONS_PATH, 1, CAPTIONS_PATH, captions))
        if not sections:
            raise ParseError(f"no text found in {pdf_path.name}")

        parsed = ParsedDocument(
            doc_id=doc_id,
            parser=f"{VERSION} (pymupdf {pymupdf.VersionBind})",
            title=report.title_from_layout,
            authors=[],
            abstract="",
            sections=sections,
            n_pages=n_pages,
            n_references=0,
        )
        parsed.issues = issues + [i for i in quality_issues(parsed) if i != "no title"]
        if statistics.mean([len(s.text.split()) for s in parsed.body_sections] or [0]) < 25:
            parsed.issues.append("very short sections: headings may be over-detected")
        return parsed, report
