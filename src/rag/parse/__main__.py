"""Debug view for the parse stage: one PDF through a parser, with everything saved.

    rag parse report.pdf                   # the default parser (PDF_PARSER, pymupdf)
    rag parse data/raw/arxiv/1706.03762.pdf --parser grobid

Prints the section tree and, for the layout parser, what it decided: where headings came from
(bookmarks or layout), the body font size, how many running headers / footers and page numbers it
dropped, the title it took from page 1. Writes data/parsed/<doc>.json (the ParsedDocument); with
GROBID also data/parsed/<doc>.tei.xml (GROBID's raw output).
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rag.config import get_settings
from rag.models import doc_file_stem, make_doc_id
from rag.parse.base import ParseError, print_parse_debug


def run(parsed_args) -> int:
    settings = get_settings()
    pdf_path: Path = parsed_args.pdf
    doc_id = parsed_args.doc_id or make_doc_id("file", pdf_path.stem)
    stem = doc_file_stem(doc_id)
    out_dir = settings.data_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    which = getattr(parsed_args, "parser", None) or settings.pdf_parser

    start = time.perf_counter()
    report = None
    try:
        if which == "grobid":
            from rag.parse.grobid import GrobidParser
            from rag.parse.tei import tei_to_parsed

            parser = GrobidParser.from_settings()
            if not parser.is_alive():
                print(f"GROBID is not running at {parser.base_url} (see AGENTS.md for the start command)")
                return 1
            tei = parser.process_pdf(pdf_path)
            parsed = tei_to_parsed(tei, doc_id, parser=f"grobid-{parser.version}")
            (out_dir / f"{stem}.tei.xml").write_text(tei, encoding="utf-8")
        else:
            from rag.parse.pdf import LayoutPDFParser

            parsed, report = LayoutPDFParser().parse_with_report(pdf_path, doc_id)
    except ParseError as exc:
        print(f"parse failed: {exc}")
        return 1
    elapsed = time.perf_counter() - start

    (out_dir / f"{stem}.json").write_text(
        json.dumps(asdict(parsed) | {"status": parsed.status}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_parse_debug(parsed)
    if report is not None:
        print(f"\nHeadings from:  {report.headings_from}")
        print(f"Title (page 1): {report.title_from_layout or '(none)'}")
        print(
            f"Dropped:        {report.dropped_header_footer} running header/footer blocks, "
            f"{report.dropped_page_numbers} page numbers"
        )
        print(f"Joined:         {report.joined_across_pages} paragraphs split by a page break")
        for note in report.notes:
            print(f"Note:           {note}")
    print(f"\n{which} time: {elapsed:.2f}s   saved: {out_dir / stem}.json")
    return 0


def main() -> int:
    from rag.main import run_module

    return run_module(["parse"])


if __name__ == "__main__":
    sys.exit(main())
