"""Domain objects shared by every stage: what a paper, a section and a chunk are.

SQLite (rag.store.db) is the source of truth for these. Search indexes only
hold chunk ids and are rebuilt from the database.

Offsets: a Section stores its full text (paragraphs joined by a blank line).
A Chunk's char_start/char_end index into that section text, so
`section.text[chunk.char_start:chunk.char_end] == chunk.text` always holds.
That is what lets a citation point at an exact span.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

ChunkType = Literal["abstract", "body", "caption", "table"]
ParseStatus = Literal["pending", "ok", "suspect", "failed"]
SourceFormat = Literal["html", "pdf"]


@dataclass
class Paper:
    arxiv_id: str  # without version suffix, e.g. "2106.09685"
    version: int
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str
    published: date
    updated: date | None = None
    source_format: SourceFormat | None = None
    source_sha256: str | None = None
    parser: str | None = None
    parse_status: ParseStatus = "pending"
    parse_error: str | None = None

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


@dataclass
class Section:
    arxiv_id: str
    ord: int  # 0-based order within the paper
    heading: str
    level: int  # 1 = top-level section, 2 = subsection, ...
    path: str  # e.g. "3 Method > 3.2 Training"; used in chunk headers and citations
    text: str
    anchor: str | None = None  # HTML id for deep links (arxiv.org/html/<id>#<anchor>)
    page_start: int | None = None  # only known for the PDF fallback

    @property
    def section_id(self) -> str:
        return section_id(self.arxiv_id, self.ord)


@dataclass
class Chunk:
    arxiv_id: str
    section_ord: int
    ord: int  # 0-based order within the paper
    chunk_type: ChunkType
    text: str
    char_start: int
    char_end: int
    n_tokens: int
    chunker_version: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.char_start < self.char_end:
            raise ValueError(
                f"bad offsets for chunk {self.ord} of {self.arxiv_id}: "
                f"char_start={self.char_start}, char_end={self.char_end}"
            )
        self.chunk_id = make_chunk_id(self.arxiv_id, self.section_ord, self.text)

    @property
    def section_id(self) -> str:
        return section_id(self.arxiv_id, self.section_ord)


def section_id(arxiv_id: str, ord: int) -> str:
    return f"{arxiv_id}:{ord}"


def make_chunk_id(arxiv_id: str, section_ord: int, text: str) -> str:
    """Content-derived id: the same text in the same section always gets the same id.

    Unlike the old `f"{doc}_{position}"`, re-chunking a paper does not shift
    the ids of chunks whose text did not change.
    """
    digest = hashlib.sha1(f"{arxiv_id}|{section_ord}|{text}".encode())
    return digest.hexdigest()[:16]
