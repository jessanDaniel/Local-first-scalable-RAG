"""PDF title guesses straight from PyMuPDF (used by ingest to choose a document's title).

The per-page fallback parser that lived here was replaced by rag.parse.pdf (LayoutPDFParser), which
finds headings and falls back to one section per page by itself.

PyMuPDF is AGPL-licensed: fine for an open-source repo; a closed-source release needs a commercial
licence or another PDF library (to decide before packaging, step 28).
"""

from pathlib import Path

import pymupdf

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
