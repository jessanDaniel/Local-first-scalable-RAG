"""Sections -> chunks.

Rules (docs/production_plan.md §4, step 3):

- Never cross a section boundary. Each chunk is one contiguous span of one
  section's text, so `section.text[chunk.char_start:chunk.char_end] == chunk.text`
  and its pages come from the paragraphs it overlaps.
- Pack whole paragraphs while the chunk stays within max_tokens, counted with
  the embedding model's tokenizer.
- A paragraph longer than max_tokens is split at sentence ends. Where a chunk
  boundary falls inside such a paragraph, the next chunk repeats the last
  sentence (one-sentence overlap), so a fact split across the boundary is
  still whole in one of the two chunks.
- A sentence longer than max_tokens (tables or formulas read as text) is cut
  at word boundaries, and a single over-long "word" at character boundaries.
- The abstract becomes "abstract" chunks; each figure/table caption becomes
  its own "caption" chunk; everything else is "body".
- The same text twice in one section is kept once (it would get the same id).

Bump CHUNKER_VERSION whenever these rules change: chunks and the indexes built
from them record which version made them.
"""

import re
from dataclasses import dataclass

from rag.chunk.tokens import TokenCounter
from rag.models import Chunk, ChunkType, Section
from rag.parse.base import ABSTRACT_PATH, CAPTIONS_PATH

CHUNKER_VERSION = "v2"
# Room left under the model limit for the contextual header added at embedding time.
HEADER_RESERVE_TOKENS = 64

SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z0-9(\[\"'])")
ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "al.",
    "fig.",
    "figs.",
    "eq.",
    "eqs.",
    "sec.",
    "secs.",
    "tab.",
    "vs.",
    "cf.",
    "resp.",
    "approx.",
    "no.",
    "ref.",
    "refs.",
    "appx.",
    "ch.",
    "pp.",
}
INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")  # "J. Smith": not a sentence end


@dataclass(frozen=True)
class _Unit:
    """A piece that is never split further: a whole paragraph, a sentence, or a hard cut."""

    start: int
    end: int
    paragraph: int
    inside_paragraph: bool  # True when the paragraph was too long and got split


class Chunker:
    def __init__(self, counter: TokenCounter, max_tokens: int):
        if max_tokens > counter.max_tokens - HEADER_RESERVE_TOKENS:
            raise ValueError(
                f"max_tokens {max_tokens} leaves no room for the chunk header: the embedding "
                f"model reads at most {counter.max_tokens} tokens"
            )
        self.counter = counter
        self.max_tokens = max_tokens
        self.version = f"{CHUNKER_VERSION}-{max_tokens}"

    def chunk_document(self, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            seen: set[str] = set()
            for start, end in self._section_spans(section):
                text = section.text[start:end]
                if text in seen:
                    continue
                seen.add(text)
                page_start, page_end = _pages(section, start, end)
                chunks.append(
                    Chunk(
                        doc_id=section.doc_id,
                        section_ord=section.ord,
                        ord=len(chunks),
                        chunk_type=_chunk_type(section),
                        text=text,
                        char_start=start,
                        char_end=end,
                        n_tokens=self.counter.count(text),
                        chunker_version=self.version,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
        return chunks

    def _section_spans(self, section: Section) -> list[tuple[int, int]]:
        units = self._units(section)
        if _chunk_type(section) == "caption":
            # One caption per chunk: pack each paragraph's units on their own.
            spans = []
            for paragraph in range(len(section.paragraphs)):
                spans.extend(self._pack(section.text, [u for u in units if u.paragraph == paragraph]))
            return spans
        return self._pack(section.text, units)

    def _units(self, section: Section) -> list[_Unit]:
        text = section.text
        units = []
        for index, paragraph in enumerate(section.paragraphs):
            if self._fits(text, paragraph.char_start, paragraph.char_end):
                units.append(_Unit(paragraph.char_start, paragraph.char_end, index, False))
                continue
            for start, end in sentence_spans(text, paragraph.char_start, paragraph.char_end):
                for piece_start, piece_end in self._hard_split(text, start, end):
                    units.append(_Unit(piece_start, piece_end, index, True))
        return units

    def _pack(self, text: str, units: list[_Unit]) -> list[tuple[int, int]]:
        spans = []
        i = 0
        while i < len(units):
            start, end = units[i].start, units[i].end
            j = i + 1
            while j < len(units) and self._fits(text, start, units[j].end):
                end = units[j].end
                j += 1
            spans.append((start, end))
            if j >= len(units):
                break
            last, following = units[j - 1], units[j]
            overlap = (
                j - 1 > i  # the repeated sentence is not the chunk's only unit, so we move forward
                and last.inside_paragraph
                and following.paragraph == last.paragraph
                and self._fits(text, last.start, following.end)
            )
            i = j - 1 if overlap else j
        return spans

    def _hard_split(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Cut [start, end) into pieces that fit, at word boundaries where possible."""
        if self._fits(text, start, end):
            return [(start, end)]
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text[start:end])]
        pieces: list[tuple[int, int]] = []
        piece_start = None
        piece_end = None
        for word_start, word_end in words:
            word_start, word_end = start + word_start, start + word_end
            if piece_start is not None and self._fits(text, piece_start, word_end):
                piece_end = word_end
                continue
            if piece_start is not None:
                pieces.append((piece_start, piece_end))
            if self._fits(text, word_start, word_end):
                piece_start, piece_end = word_start, word_end
            else:
                pieces.extend(self._split_characters(text, word_start, word_end))
                piece_start = piece_end = None
        if piece_start is not None:
            pieces.append((piece_start, piece_end))
        return pieces

    def _split_characters(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        pieces = []
        while start < end:
            low, high = start + 1, end
            while low < high:  # longest prefix that fits
                middle = (low + high + 1) // 2
                if self._fits(text, start, middle):
                    low = middle
                else:
                    high = middle - 1
            pieces.append((start, low))
            start = low
        return pieces

    def _fits(self, text: str, start: int, end: int) -> bool:
        return self.counter.count(text[start:end]) <= self.max_tokens


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence offsets within text[start:end], skipping abbreviations and initials."""
    spans = []
    sentence_start = start
    for match in SENTENCE_END.finditer(text, start, end):
        sentence_end = match.start() + len(match.group().rstrip())
        before = text[sentence_start : match.start() + 1]
        last_word = before.split()[-1].lstrip("([\"'").lower() if before.split() else ""
        if last_word in ABBREVIATIONS or INITIAL.search(before):
            continue
        spans.append((sentence_start, sentence_end))
        sentence_start = match.end()
    if sentence_start < end:
        spans.append((sentence_start, end))
    return spans


def _chunk_type(section: Section) -> ChunkType:
    if section.path == ABSTRACT_PATH:
        return "abstract"
    if section.path == CAPTIONS_PATH:
        return "caption"
    return "body"


def _pages(section: Section, start: int, end: int) -> tuple[int | None, int | None]:
    starts = [
        p.page_start
        for p in section.paragraphs
        if p.char_start < end and p.char_end > start and p.page_start is not None
    ]
    ends = [
        p.page_end for p in section.paragraphs if p.char_start < end and p.char_end > start and p.page_end is not None
    ]
    return (min(starts) if starts else None, max(ends) if ends else None)
