"""Plain text and Markdown -> ParsedDocument. No external service, no pages.

Markdown: `#` headings make the section tree (`# A` > `## B` gives path "A > B"). A
single top-level heading that opens the file is the title, not a section.

Plain text has no markup, so headings are guessed, and the guess fails safe:
a block is a heading only if it is ONE line, at most 12 words and 80
characters, starts with a capital letter or a digit, does not end in sentence
punctuation, and is followed by more text. Anything else is a paragraph. The
first line of the file is the title when it passes the same test.

Paragraphs are blocks separated by blank lines. Hard-wrapped lines inside a
paragraph are joined with spaces; list items and fenced code keep their line
breaks. Text before the first heading goes in a "Preamble" section.
"""

import re
from pathlib import Path

from rag.models import Section
from rag.parse.base import ParsedDocument, ParseError, Parser, quality_issues

TEXT_SUFFIXES = {".txt": "txt", ".text": "txt", ".md": "md", ".markdown": "md"}
PREAMBLE = "Preamble"
MAX_HEADING_WORDS = 12
MAX_HEADING_CHARS = 80
SENTENCE_END = tuple(".,;:!?")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
FENCE = re.compile(r"^\s*(```|~~~)")
PATH_SEPARATOR = " > "


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    return (
        bool(line)
        and len(line) <= MAX_HEADING_CHARS
        and len(line.split()) <= MAX_HEADING_WORDS
        and (line[0].isupper() or line[0].isdigit())
        and not line.endswith(SENTENCE_END)
    )


SHORT_LINE_CHARS = 60  # hard-wrapped prose fills its lines; a block of only short lines is a list of facts


def join_block(lines: list[str]) -> str:
    """One paragraph's text: wrapped prose joined with spaces; lists, code and short-line blocks
    ("Effective Date: ...", "Version: 3.2") keep their line breaks."""
    stripped = [line.strip() for line in lines]
    continues = any(nxt[:1].islower() for nxt in stripped[1:])  # a wrapped sentence goes on in lowercase
    short_lines = len(lines) > 1 and not continues and all(len(line) < SHORT_LINE_CHARS for line in stripped)
    if short_lines or any(LIST_ITEM.match(line) for line in lines) or (lines and FENCE.match(lines[0])):
        return "\n".join(line.rstrip() for line in lines)
    return " ".join(" ".join(line.split()) for line in lines)


def blocks(text: str) -> list[list[str]]:
    """Blank-line separated blocks of lines. A fenced code block stays one block even with blank lines."""
    result: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                result.append(current)
                current = []
            continue
        current.append(line)
    if current:
        result.append(current)
    return result


class TextParser(Parser):
    name = "text-v1"

    def parse(self, path: Path, doc_id: str) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ParseError(f"cannot read {path.name} as UTF-8 text: {exc}") from exc
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ParseError(f"{path.name} is empty")
        markdown = TEXT_SUFFIXES.get(path.suffix.lower()) == "md"
        title, sections = (self._markdown if markdown else self._plain)(text, doc_id)
        if not sections:
            raise ParseError(f"no text found in {path.name}")
        parsed = ParsedDocument(
            doc_id=doc_id,
            parser=self.name,
            title=title,
            authors=[],
            abstract="",
            sections=sections,
            n_pages=None,
            n_references=0,
        )
        parsed.issues = quality_issues(parsed)
        return parsed

    # --- plain text -----------------------------------------------------------

    def _plain(self, text: str, doc_id: str) -> tuple[str, list[Section]]:
        all_blocks = blocks(text)
        title = ""
        builder = _SectionBuilder(doc_id)
        first = all_blocks[0]
        if looks_like_heading(first[0]):
            title = first[0].strip()
            all_blocks.pop(0)
            # Lines under the title in the same block ("Version: 3.2") are facts about the
            # document, never a heading, even when one of them looks like one.
            if first[1:]:
                builder.paragraph(join_block(first[1:]))

        for i, block in enumerate(all_blocks):
            followed_by_text = i + 1 < len(all_blocks)
            if len(block) == 1 and looks_like_heading(block[0]) and followed_by_text:
                builder.heading(block[0].strip(), level=1)
            else:
                builder.paragraph(join_block(block))
        return title, builder.finish()

    # --- Markdown -------------------------------------------------------------

    def _markdown(self, text: str, doc_id: str) -> tuple[str, list[Section]]:
        # Split heading lines out of their blocks: a Markdown heading needs no blank line after it.
        items: list[tuple[str, int, str]] = []  # ("heading", level, text) or ("paragraph", 0, text)
        for block in blocks(text):
            pending: list[str] = []
            in_fence = False
            for line in block:
                if FENCE.match(line):
                    in_fence = not in_fence
                match = None if in_fence else MD_HEADING.match(line)
                if match:
                    if pending:
                        items.append(("paragraph", 0, join_block(pending)))
                        pending = []
                    items.append(("heading", len(match.group(1)), match.group(2).strip()))
                else:
                    pending.append(line)
            if pending:
                items.append(("paragraph", 0, join_block(pending)))

        title = ""
        top_levels = [level for kind, level, _ in items if kind == "heading" and level == 1]
        if items and items[0][0] == "heading" and items[0][1] == 1 and len(top_levels) == 1:
            title = items.pop(0)[2]  # "# Title" opening the file names the document
            items = [(kind, level - 1 if kind == "heading" else 0, value) for kind, level, value in items]

        builder = _SectionBuilder(doc_id)
        for kind, level, value in items:
            if kind == "heading":
                builder.heading(value, max(level, 1))
            else:
                builder.paragraph(value)
        return title, builder.finish()


class _SectionBuilder:
    """Collects headings and paragraphs into Sections with "A > B" paths. Empty sections are dropped."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.sections: list[Section] = []
        self.stack: list[tuple[int, str]] = []  # (level, heading) of the current path
        self.paragraphs: list[tuple[str, int | None, int | None]] = []  # (text, page_start, page_end)
        self.current = (PREAMBLE, 1, PREAMBLE)  # heading, level, path

    def heading(self, text: str, level: int) -> None:
        self._flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        self.current = (text, len(self.stack), PATH_SEPARATOR.join(h for _, h in self.stack))

    def paragraph(self, text: str, page_start: int | None = None, page_end: int | None = None) -> None:
        if text.strip():
            self.paragraphs.append((text, page_start, page_end))

    def _flush(self) -> None:
        if self.paragraphs:
            heading, level, path = self.current
            self.sections.append(
                Section.from_paragraphs(self.doc_id, len(self.sections), heading, level, path, self.paragraphs)
            )
        self.paragraphs = []

    def finish(self) -> list[Section]:
        self._flush()
        return self.sections
