"""What the UI needs to show a source: the stored file (a PDF opened at a page) and the passage in
its section (any format).

Only documents the database knows can be served, and only from the folders ingest writes to
(data/raw/arxiv, data/raw/files): the path is built from the database record, never from the
request, and checked to stay inside data/raw after resolving links.
"""

from pathlib import Path

from rag.models import Document, doc_file_stem
from rag.store import db

MEDIA_TYPES = {"pdf": "application/pdf", "txt": "text/plain; charset=utf-8", "md": "text/plain; charset=utf-8"}


def stored_file(data_dir: Path, document: Document) -> Path | None:
    """The file ingest saved for this document, or None (unknown source, missing, or outside data/raw)."""
    raw = (data_dir / "raw").resolve()
    if document.source == "arxiv":
        arxiv_id = str(document.metadata.get("arxiv_id") or document.doc_id.split(":", 1)[1])
        path = data_dir / "raw" / "arxiv" / f"{arxiv_id.replace('/', '_')}.pdf"
    elif document.source == "file" and document.source_format in MEDIA_TYPES:
        path = data_dir / "raw" / "files" / f"{doc_file_stem(document.doc_id)}.{document.source_format}"
    else:
        return None
    resolved = path.resolve()
    if not resolved.is_relative_to(raw) or not resolved.is_file():
        return None
    return resolved


def file_format(document: Document) -> str | None:
    return "pdf" if document.source == "arxiv" else document.source_format


def document_info(conn, data_dir: Path, document: Document) -> dict:
    path = stored_file(data_dir, document)
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "authors": document.authors,
        "collection": document.collection_id,
        "format": file_format(document),
        "has_file": path is not None,
        "file_url": f"/documents/{document.doc_id}/file" if path is not None else None,
        "source_link": document.source_uri if (document.source_uri or "").startswith("https://") else None,
        "n_sections": len(db.get_sections(conn, document.doc_id)),
    }


def passage_context(conn, data_dir: Path, chunk_ids: list[str]) -> dict | None:
    """The sections holding these chunks (one document), with each chunk's character range to
    highlight. None if a chunk is unknown or the chunks span documents."""
    chunks = db.get_chunks_by_ids(conn, chunk_ids)
    if len(chunks) != len(set(chunk_ids)) or len({c.doc_id for c in chunks}) != 1:
        return None
    doc_id = chunks[0].doc_id
    document = db.get_document(conn, doc_id)
    sections = {s.ord: s for s in db.get_sections(conn, doc_id)}
    out = []
    for ord_ in sorted({c.section_ord for c in chunks}):
        section = sections[ord_]
        out.append(
            {
                "ord": ord_,
                "path": section.path,
                "page_start": section.page_start,
                "text": section.text,
                "highlights": [
                    {"start": c.char_start, "end": c.char_end, "chunk_id": c.chunk_id, "page": c.page_start}
                    for c in sorted(chunks, key=lambda c: c.char_start)
                    if c.section_ord == ord_
                ],
            }
        )
    return {"document": document_info(conn, data_dir, document), "sections": out}
