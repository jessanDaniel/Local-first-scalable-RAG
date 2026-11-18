"""Ingest papers: arXiv metadata -> PDF -> parse -> SQLite.

    uv run python -m rag.ingest corpus/seed_ai_research.txt --limit 10 --report <csv>

Per paper: metadata from arXiv (title, authors, abstract, dates are taken
from here, not from the PDF), PDF download (skipped if already on disk),
GROBID parse, PyMuPDF if GROBID fails, then the paper and its sections are
written to the database with status ok / suspect / failed. Papers already
parsed are skipped unless --reparse. Each parse is also dumped to
data/parsed/<id>.json (and GROBID's TEI to <id>.tei.xml) for inspection.
"""

import argparse
import csv
import json
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rag.acquire.arxiv import ArxivClient, ArxivError, read_id_file
from rag.config import Settings, get_settings
from rag.models import Paper
from rag.parse.base import ParsedPaper, ParseError, Parser, ParserUnavailableError
from rag.parse.grobid import GrobidParser
from rag.parse.pymupdf import PyMuPDFParser
from rag.store import db


@dataclass
class IngestResult:
    arxiv_id: str
    title: str
    status: str  # ok | suspect | failed | skipped
    parser: str
    used_fallback: bool
    downloaded: bool
    download_seconds: float
    parse_seconds: float
    n_pages: int | None
    n_sections: int
    body_words: int
    issues: str


def ingest_paper(
    paper: Paper,
    conn: sqlite3.Connection,
    client: ArxivClient,
    primary: Parser,
    fallback: Parser,
    pdf_dir: Path,
    parsed_dir: Path,
) -> IngestResult:
    start = time.perf_counter()
    pdf_path, sha256, downloaded = client.download_pdf(paper, pdf_dir)
    download_seconds = time.perf_counter() - start

    start = time.perf_counter()
    used_fallback = False
    errors: list[str] = []
    parsed: ParsedPaper | None = None
    for parser in (primary, fallback):
        try:
            parsed = parser.parse(pdf_path, paper.arxiv_id)
            break
        except ParserUnavailableError:
            raise
        except ParseError as exc:
            errors.append(f"{parser.name}: {exc}")
            used_fallback = True
    parse_seconds = time.perf_counter() - start

    if parsed is None:
        db.upsert_paper(conn, replace(paper, source_sha256=sha256, parse_status="failed", parse_error=" | ".join(errors)))
        db.replace_sections(conn, paper.arxiv_id, [])
        return IngestResult(
            paper.arxiv_id, paper.title, "failed", "", used_fallback, downloaded,
            download_seconds, parse_seconds, None, 0, 0, " | ".join(errors),
        )

    # The PDF parser only reads the body; metadata issues it reports are covered by arXiv.
    issues = [i for i in parsed.issues if i not in ("no title", "no abstract")] + errors
    status = "suspect" if issues else "ok"
    db.upsert_paper(
        conn,
        replace(
            paper,
            source_sha256=sha256,
            parser=parsed.parser,
            parse_status=status,
            parse_error="; ".join(issues) or None,
        ),
    )
    db.replace_sections(conn, paper.arxiv_id, parsed.sections)

    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{paper.arxiv_id}.json").write_text(
        json.dumps(asdict(parsed) | {"status": status, "issues": issues}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return IngestResult(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        status=status,
        parser=parsed.parser,
        used_fallback=used_fallback,
        downloaded=downloaded,
        download_seconds=download_seconds,
        parse_seconds=parse_seconds,
        n_pages=parsed.n_pages,
        n_sections=len(parsed.body_sections),
        body_words=sum(len(s.text.split()) for s in parsed.body_sections),
        issues="; ".join(issues),
    )


def ingest(
    arxiv_ids: list[str],
    settings: Settings,
    reparse: bool = False,
) -> list[IngestResult]:
    pdf_dir = settings.data_dir / "raw" / "pdf"
    parsed_dir = settings.data_dir / "parsed"
    conn = db.open_db(settings.db_path)
    client = ArxivClient(settings)
    primary = GrobidParser.from_settings(settings, tei_dir=parsed_dir)
    fallback = PyMuPDFParser()

    results: list[IngestResult] = []
    todo = []
    for arxiv_id in arxiv_ids:
        existing = db.get_paper(conn, arxiv_id)
        if existing and existing.parse_status != "pending" and not reparse:
            results.append(IngestResult(
                arxiv_id, existing.title, "skipped", existing.parser or "", False, False,
                0.0, 0.0, None, 0, 0, f"already {existing.parse_status}",
            ))
        else:
            todo.append(arxiv_id)
    if not todo:
        return results

    if not primary.is_alive():
        raise SystemExit(f"GROBID is not running at {primary.base_url}; see AGENTS.md for the start command")

    for paper in client.fetch_metadata(todo):
        db.upsert_paper(conn, paper)  # recorded as pending before any download
        result = ingest_paper(paper, conn, client, primary, fallback, pdf_dir, parsed_dir)
        print(
            f"{result.status:8} {result.arxiv_id:11} {result.parse_seconds:6.2f}s "
            f"{result.n_sections:3} sections {result.body_words:6} words  {result.title[:60]}",
            flush=True,
        )
        results.append(result)
    conn.close()
    return results


def print_summary(results: list[IngestResult]) -> None:
    done = [r for r in results if r.status != "skipped"]
    print(f"\n{len(results)} papers: " + ", ".join(
        f"{status} {sum(r.status == status for r in results)}"
        for status in ("ok", "suspect", "failed", "skipped")
        if any(r.status == status for r in results)
    ))
    if not done:
        return
    fallback = sum(r.used_fallback for r in done)
    print(f"fallback used: {fallback}/{len(done)} ({fallback / len(done):.0%})")
    parse_times = [r.parse_seconds for r in done if r.status != "failed"]
    if parse_times:
        print(
            f"parse seconds: median {statistics.median(parse_times):.2f}, "
            f"max {max(parse_times):.2f}, total {sum(parse_times):.1f}"
        )
    print(f"downloaded: {sum(r.downloaded for r in done)} new PDFs, "
          f"{sum(r.download_seconds for r in done):.1f}s including throttling")
    for r in done:
        if r.issues:
            print(f"  {r.arxiv_id}: {r.issues}")


def main() -> int:
    args = argparse.ArgumentParser(description="Ingest arXiv papers into the SQLite store.")
    args.add_argument("id_file", type=Path, help="one arXiv id per line, '#' starts a comment")
    args.add_argument("--limit", type=int, help="only the first N ids")
    args.add_argument("--reparse", action="store_true", help="parse again even if already parsed")
    args.add_argument("--report", type=Path, help="write per-paper results to this CSV")
    parsed_args = args.parse_args()

    ids = read_id_file(parsed_args.id_file)[: parsed_args.limit]
    try:
        results = ingest(ids, get_settings(), reparse=parsed_args.reparse)
    except ArxivError as exc:
        print(f"arXiv error: {exc}")
        return 1
    except ParserUnavailableError as exc:
        print(f"stopped: {exc}. Papers done so far are saved; rerun to continue.")
        return 1
    print_summary(results)

    if parsed_args.report:
        parsed_args.report.parent.mkdir(parents=True, exist_ok=True)
        with parsed_args.report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r) | {
                    "download_seconds": round(r.download_seconds, 2),
                    "parse_seconds": round(r.parse_seconds, 2),
                })
        print(f"report: {parsed_args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
