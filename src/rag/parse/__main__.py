"""Debug view for the parse stage: one PDF through GROBID, with everything saved.

    uv run python -m rag.parse data/raw/pdf/2305.14314.pdf

Writes data/parsed/<doc>.tei.xml (GROBID's raw output) and data/parsed/<doc>.json
(the ParsedDocument), then prints the section tree. GROBID must be running.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rag.config import get_settings
from rag.models import doc_file_stem, make_doc_id
from rag.parse.base import ParseError, print_parse_debug
from rag.parse.grobid import GrobidParser
from rag.parse.tei import tei_to_parsed


def main() -> int:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("pdf", type=Path)
    args.add_argument("--doc-id", help='defaults to "file:<pdf name without .pdf>"')
    parsed_args = args.parse_args()

    pdf_path: Path = parsed_args.pdf
    doc_id = parsed_args.doc_id or make_doc_id("file", pdf_path.stem)
    stem = doc_file_stem(doc_id)
    out_dir = get_settings().data_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = GrobidParser.from_settings()
    if not parser.is_alive():
        print(f"GROBID is not running at {parser.base_url} (see AGENTS.md for the start command)")
        return 1

    start = time.perf_counter()
    try:
        tei = parser.process_pdf(pdf_path)
    except ParseError as exc:
        print(f"parse failed: {exc}")
        return 1
    elapsed = time.perf_counter() - start

    parsed = tei_to_parsed(tei, doc_id, parser=f"grobid-{parser.version}")
    (out_dir / f"{stem}.tei.xml").write_text(tei, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(asdict(parsed) | {"status": parsed.status}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_parse_debug(parsed)
    print(f"\nGROBID time: {elapsed:.2f}s   saved: {out_dir / stem}.tei.xml, .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
