"""Domain objects shared by every stage: what a document, a section and a chunk are.

SQLite (rag.store.db) is the source of truth for these. Search indexes only
hold chunk ids and are rebuilt from the database.

Offsets: a Section stores its full text (paragraphs joined by a blank line)
and, per paragraph, its offsets and PDF pages. A Chunk's char_start/char_end
index into that section text, so
`section.text[chunk.char_start:chunk.char_end] == chunk.text` always holds,
and the paragraphs a chunk overlaps give its page range. That is what lets a
citation point at an exact span and page.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

ChunkType = Literal["abstract", "body", "caption", "table"]
ParseStatus = Literal["pending", "ok", "suspect", "failed"]
SourceType = Literal["arxiv", "file"]
SourceFormat = Literal["html", "pdf", "txt", "md"]

# A collection groups documents that are searched together (and, in Stage 5, share access rules).
# Each document belongs to exactly one. Ids are slugs: "papers", "hr-demo".
COLLECTION_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
DEFAULT_COLLECTION = "default"


def check_collection_id(collection_id: str) -> str:
    if not COLLECTION_ID.fullmatch(collection_id):
        raise ValueError(
            f"collection {collection_id!r}: use lowercase letters, digits and hyphens, starting with a letter or digit"
        )
    return collection_id


@dataclass
class Document:
    """One ingested document, whatever it is: an arXiv paper, a local PDF, later a web page.

    doc_id is "<source>:<key>" ("arxiv:1412.6980", "file:perceptron-1958"), so documents from
    different sources never collide. Fields every source can fill live here; anything only
    one source has (arXiv version and categories, ...) goes in `metadata`.
    """

    doc_id: str
    source: SourceType
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""  # the source's own summary, when it has one
    published: date | None = None
    source_uri: str | None = None  # where a reader opens it: arXiv abs URL, or the local file path
    source_format: SourceFormat | None = None
    source_sha256: str | None = None
    metadata: dict = field(default_factory=dict)
    parser: str | None = None
    parse_status: ParseStatus = "pending"
    parse_error: str | None = None
    collection_id: str = DEFAULT_COLLECTION

    def __post_init__(self) -> None:
        if not self.doc_id.startswith(f"{self.source}:") or self.doc_id == f"{self.source}:":
            raise ValueError(f"doc_id {self.doc_id!r} must look like '{self.source}:<key>'")
        check_collection_id(self.collection_id)

    @property
    def key(self) -> str:
        """The part after "<source>:", e.g. the arXiv id."""
        return self.doc_id.split(":", 1)[1]


DOC_KEY = re.compile(r"[^a-z0-9._-]+")


def doc_file_stem(doc_id: str) -> str:
    """File-name-safe form of a doc_id for files written per document: "arxiv:1412.6980" -> "arxiv_1412.6980"."""
    return doc_id.replace(":", "_").replace("/", "_")


def make_doc_id(source: SourceType, key: str) -> str:
    """Build a doc_id. arXiv keys are kept as is; other keys are slugged:
    ("file", "The Perceptron (1958)") -> "file:the-perceptron-1958"."""
    if source != "arxiv":
        key = DOC_KEY.sub("-", key.lower()).strip("-.")
    if not key:
        raise ValueError("empty document key")
    return f"{source}:{key}"


@dataclass(frozen=True)
class Paragraph:
    """Where one paragraph sits in its section's text, and which PDF pages it spans."""

    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None


PARAGRAPH_SEPARATOR = "\n\n"


@dataclass
class Section:
    doc_id: str
    ord: int  # 0-based order within the document
    heading: str
    level: int  # 1 = top-level section, 2 = subsection, ...
    path: str  # e.g. "3 Method > 3.2 Training"; used in chunk headers and citations
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    anchor: str | None = None  # optional id for deep links into the source document
    page_start: int | None = None

    def __post_init__(self) -> None:
        previous_end = 0
        for paragraph in self.paragraphs:
            if not previous_end <= paragraph.char_start < paragraph.char_end <= len(self.text):
                raise ValueError(
                    f"paragraph offsets out of order or outside the text in section {self.section_id}: {paragraph}"
                )
            previous_end = paragraph.char_end

    @property
    def section_id(self) -> str:
        return section_id(self.doc_id, self.ord)

    @classmethod
    def from_paragraphs(
        cls,
        doc_id: str,
        ord: int,
        heading: str,
        level: int,
        path: str,
        paragraphs: list[tuple[str, int | None, int | None]],
    ) -> "Section":
        """Join (text, page_start, page_end) paragraphs into one section text, recording offsets."""
        texts: list[str] = []
        spans: list[Paragraph] = []
        position = 0
        for text, page_start, page_end in paragraphs:
            if texts:
                position += len(PARAGRAPH_SEPARATOR)
            spans.append(Paragraph(position, position + len(text), page_start, page_end))
            texts.append(text)
            position += len(text)
        pages = [span.page_start for span in spans if span.page_start is not None]
        return cls(
            doc_id=doc_id,
            ord=ord,
            heading=heading,
            level=level,
            path=path,
            text=PARAGRAPH_SEPARATOR.join(texts),
            paragraphs=spans,
            page_start=min(pages) if pages else None,
        )


@dataclass
class Chunk:
    doc_id: str
    section_ord: int
    ord: int  # 0-based order within the document
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
                f"bad offsets for chunk {self.ord} of {self.doc_id}: "
                f"char_start={self.char_start}, char_end={self.char_end}"
            )
        self.chunk_id = make_chunk_id(self.doc_id, self.section_ord, self.text)

    @property
    def section_id(self) -> str:
        return section_id(self.doc_id, self.section_ord)


def section_id(doc_id: str, ord: int) -> str:
    return f"{doc_id}:{ord}"


def make_chunk_id(doc_id: str, section_ord: int, text: str) -> str:
    """Content-derived id: the same text in the same section always gets the same id.

    Unlike the old `f"{doc}_{position}"`, re-chunking a document does not shift
    the ids of chunks whose text did not change.
    """
    digest = hashlib.sha1(f"{doc_id}|{section_ord}|{text}".encode())
    return digest.hexdigest()[:16]
