"""Phase 1 — Document ingestion.

Turns a raw file into extracted text plus basic, inspectable stats.

Deliberately does NOT do chunking here (see src/ingestion/chunker.py later) —
ingestion and chunking are separate concerns with separate failure modes.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    document_id: str
    source_path: str
    text: str
    char_count: int
    word_count: int


def load_txt(path: str | Path) -> LoadedDocument:
    """Load a single .txt file and return its text plus basic stats.

    Steps to implement:
        1. Turn `path` into a `Path` object (if it isn't already).
        2. Read the file's full text contents (UTF-8).
        3. Derive `document_id` from the filename, without its extension.
           Hint: Path has a `.stem` property for exactly this.
        4. Compute `char_count` — the length of the text.
        5. Compute `word_count` — split the text on whitespace and count
           the pieces.
        6. Construct and return a `LoadedDocument` with all fields set.

    Raises:
        FileNotFoundError: if `path` does not exist (this happens for free
        if you just try to open/read a missing file — no need to check
        existence yourself first).
    """
    resolved_path = Path(path)
    text = resolved_path.read_text(encoding="utf-8")

    document_id = resolved_path.stem
    char_count = len(text)
    word_count = len(text.split())

    return LoadedDocument(
        document_id=document_id,
        source_path=str(resolved_path),
        text=text,
        char_count=char_count,
        word_count=word_count,
    )
