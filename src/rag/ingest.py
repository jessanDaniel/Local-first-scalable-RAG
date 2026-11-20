"""Ingest documents: source -> PDF -> parse -> SQLite.

    rag ingest arxiv corpus/seed_ai_research.txt [--limit N]
    rag ingest files data/raw/manual/            # or individual .pdf paths
    rag ingest tei      # re-apply TEI rules to saved GROBID output, no GROBID

Two sources so far. `arxiv`: metadata from the arXiv API (title, authors,
abstract, dates come from there, not from the PDF), PDF downloaded to
data/raw/arxiv/. `files`: local PDFs, copied to data/raw/files/; title and
authors come from the parse, or the file name when the parse has none.

Every document then goes the same way: GROBID, PyMuPDF if GROBID fails,
document + sections written to the database as ok / suspect / failed.
Documents already parsed are skipped unless --reparse (a local file whose
content changed is always parsed again). Each parse is also dumped to
data/parsed/<doc>.json, and GROBID's TEI to <doc>.tei.xml, for inspection.
"""

import csv
import hashlib
import json
import shutil
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rag.acquire.arxiv import ArxivClient, ArxivError, read_id_file
from rag.config import Settings, get_settings
from rag.models import Document, doc_file_stem, make_doc_id
from rag.parse.base import ParsedDocument, ParseError, Parser, ParserUnavailableError
from rag.parse.grobid import GrobidParser
from rag.parse.pymupdf import PyMuPDFParser
from rag.parse.tei import tei_to_parsed
from rag.store import db

# The PDF parser reads the body; this issue is covered by the source's metadata or the file name.
METADATA_ISSUES = {"no title"}


@dataclass
class IngestResult:
    doc_id: str
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


def skipped(document: Document) -> IngestResult:
    return IngestResult(
        document.doc_id,
        document.title,
        "skipped",
        document.parser or "",
        False,
        False,
        0.0,
        0.0,
        None,
        0,
        0,
        f"already {document.parse_status}",
    )


def parse_and_store(
    document: Document,
    pdf_path: Path,
    conn: sqlite3.Connection,
    primary: Parser,
    fallback: Parser,
    parsed_dir: Path,
    downloaded: bool = False,
    download_seconds: float = 0.0,
) -> IngestResult:
    start = time.perf_counter()
    used_fallback = False
    errors: list[str] = []
    parsed: ParsedDocument | None = None
    for parser in (primary, fallback):
        try:
            parsed = parser.parse(pdf_path, document.doc_id)
            break
        except ParserUnavailableError:
            raise
        except ParseError as exc:
            errors.append(f"{parser.name}: {exc}")
            used_fallback = True
    parse_seconds = time.perf_counter() - start

    if parsed is None:
        error = " | ".join(errors)
        db.upsert_document(
            conn, replace(document, title=document.title or pdf_path.stem, parse_status="failed", parse_error=error)
        )
        db.replace_sections(conn, document.doc_id, [])
        return IngestResult(
            document.doc_id,
            document.title,
            "failed",
            "",
            used_fallback,
            downloaded,
            download_seconds,
            parse_seconds,
            None,
            0,
            0,
            error,
        )

    document, status, issues = store_parsed(conn, document, parsed, errors, parsed_dir, pdf_path.stem)
    return IngestResult(
        doc_id=document.doc_id,
        title=document.title,
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


def store_parsed(
    conn: sqlite3.Connection,
    document: Document,
    parsed: ParsedDocument,
    errors: list[str],
    parsed_dir: Path,
    fallback_title: str,
) -> tuple[Document, str, list[str]]:
    """Write a parse result: document row, its sections (old chunks go with them), JSON dump."""
    issues = [i for i in parsed.issues if i not in METADATA_ISSUES] + errors
    status = "suspect" if issues else "ok"
    document = replace(
        document,
        # Source metadata wins; the parse fills only what the source does not know.
        title=document.title or parsed.title or fallback_title,
        authors=document.authors or parsed.authors,
        abstract=document.abstract or parsed.abstract,
        parser=parsed.parser,
        parse_status=status,
        parse_error="; ".join(issues) or None,
    )
    db.upsert_document(conn, document)
    db.replace_sections(conn, document.doc_id, parsed.sections)

    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{doc_file_stem(document.doc_id)}.json").write_text(
        json.dumps(asdict(parsed) | {"status": status, "issues": issues}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return document, status, issues


class Ingestor:
    """What every source shares: the database, the parsers, the output folders."""

    def __init__(self, settings: Settings, reparse: bool = False):
        self.settings = settings
        self.reparse = reparse
        self.parsed_dir = settings.data_dir / "parsed"
        self.conn = db.open_db(settings.db_path)
        self.primary = GrobidParser.from_settings(settings, tei_dir=self.parsed_dir)
        self.fallback = PyMuPDFParser()
        self._checked_grobid = False

    def close(self) -> None:
        self.conn.close()

    def already_parsed(self, doc_id: str, sha256: str | None = None) -> Document | None:
        """The stored document when it can be skipped, else None."""
        existing = db.get_document(self.conn, doc_id)
        if existing is None or existing.parse_status == "pending" or self.reparse:
            return None
        if sha256 is not None and existing.source_sha256 != sha256:
            return None  # the file changed since it was parsed
        return existing

    def parse(self, document: Document, pdf_path: Path, **timing) -> IngestResult:
        if not self._checked_grobid:
            if not self.primary.is_alive():
                raise ParserUnavailableError(
                    f"GROBID is not running at {self.primary.base_url}; see AGENTS.md for the start command"
                )
            self._checked_grobid = True
        db.upsert_document(self.conn, replace(document, title=document.title or pdf_path.stem))
        result = parse_and_store(document, pdf_path, self.conn, self.primary, self.fallback, self.parsed_dir, **timing)
        print(
            f"{result.status:8} {result.doc_id:26} {result.parse_seconds:6.2f}s "
            f"{result.n_sections:3} sections {result.body_words:6} words  {result.title[:50]}",
            flush=True,
        )
        return result

    def ingest_arxiv(self, arxiv_ids: list[str]) -> list[IngestResult]:
        results, todo = [], []
        for arxiv_id in arxiv_ids:
            existing = self.already_parsed(make_doc_id("arxiv", arxiv_id))
            if existing:
                results.append(skipped(existing))
            else:
                todo.append(arxiv_id)
        if not todo:
            return results

        client = ArxivClient(self.settings)
        pdf_dir = self.settings.data_dir / "raw" / "arxiv"
        for document in client.fetch_metadata(todo):
            start = time.perf_counter()
            pdf_path, sha256, downloaded = client.download_pdf(document, pdf_dir)
            document = replace(document, source_sha256=sha256)
            results.append(
                self.parse(
                    document,
                    pdf_path,
                    downloaded=downloaded,
                    download_seconds=time.perf_counter() - start,
                )
            )
        return results

    def reapply_tei(self) -> list[IngestResult]:
        """Run the current TEI rules again on GROBID output saved by earlier runs. No GROBID needed.

        Use after changing rag/parse/tei.py. Replaces sections (and so drops chunks: rechunk after).
        """
        results = []
        for document in db.list_documents(self.conn):
            if not (document.parser or "").startswith("grobid"):
                continue
            tei_path = self.parsed_dir / f"{doc_file_stem(document.doc_id)}.tei.xml"
            if not tei_path.exists():
                results.append(replace(skipped(document), issues="no saved TEI; use --reparse with GROBID"))
                continue
            start = time.perf_counter()
            parsed = tei_to_parsed(tei_path.read_text(encoding="utf-8"), document.doc_id, document.parser)
            document, status, issues = store_parsed(self.conn, document, parsed, [], self.parsed_dir, document.title)
            results.append(
                IngestResult(
                    document.doc_id,
                    document.title,
                    status,
                    parsed.parser,
                    False,
                    False,
                    0.0,
                    time.perf_counter() - start,
                    parsed.n_pages,
                    len(parsed.body_sections),
                    sum(len(s.text.split()) for s in parsed.body_sections),
                    "; ".join(issues),
                )
            )
        return results

    def ingest_files(self, paths: list[Path]) -> list[IngestResult]:
        files_dir = self.settings.data_dir / "raw" / "files"
        results = []
        for path in paths:
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"{path}: only PDF files are supported so far")
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            doc_id = make_doc_id("file", path.stem)
            original = str(path.resolve())

            stored = db.get_document(self.conn, doc_id)
            if stored is not None and stored.metadata.get("original_path") != original:
                raise ValueError(
                    f"{path}: {doc_id} is already used by {stored.metadata.get('original_path')}; rename this file"
                )
            existing = self.already_parsed(doc_id, sha256)
            if existing:
                results.append(skipped(existing))
                continue

            # Keep our own copy, so moving or deleting the original does not break citations.
            files_dir.mkdir(parents=True, exist_ok=True)
            copy = files_dir / f"{doc_file_stem(doc_id)}.pdf"
            shutil.copyfile(path, copy)
            document = Document(
                doc_id=doc_id,
                source="file",
                title="",
                source_uri=str(copy),
                source_format="pdf",
                source_sha256=sha256,
                metadata={"original_path": original},
            )
            results.append(self.parse(document, copy))
        return results


def print_summary(results: list[IngestResult]) -> None:
    done = [r for r in results if r.status != "skipped"]
    print(
        f"\n{len(results)} documents: "
        + ", ".join(
            f"{status} {sum(r.status == status for r in results)}"
            for status in ("ok", "suspect", "failed", "skipped")
            if any(r.status == status for r in results)
        )
    )
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
    print(
        f"downloaded: {sum(r.downloaded for r in done)} new PDFs, "
        f"{sum(r.download_seconds for r in done):.1f}s including throttling"
    )
    for r in done:
        if r.issues:
            print(f"  {r.doc_id}: {r.issues}")


def write_report(results: list[IngestResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for r in results:
            writer.writerow(
                asdict(r)
                | {
                    "download_seconds": round(r.download_seconds, 2),
                    "parse_seconds": round(r.parse_seconds, 2),
                }
            )
    print(f"report: {path}")


def pdf_paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for item in inputs:
        paths.extend(sorted(item.glob("*.pdf")) if item.is_dir() else [item])
    return paths


def run(parsed_args) -> int:

    ingestor = Ingestor(get_settings(), reparse=parsed_args.reparse)
    try:
        if parsed_args.source == "arxiv":
            results = ingestor.ingest_arxiv(read_id_file(parsed_args.id_file)[: parsed_args.limit])
        elif parsed_args.source == "files":
            results = ingestor.ingest_files(pdf_paths(parsed_args.paths))
        else:
            results = ingestor.reapply_tei()
    except ArxivError as exc:
        print(f"arXiv error: {exc}")
        return 1
    except ParserUnavailableError as exc:
        print(f"stopped: {exc}. Documents done so far are saved; rerun to continue.")
        return 1
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        ingestor.close()

    print_summary(results)
    if any(r.status != "skipped" for r in results):
        print("sections replaced: run `rag chunk` to rebuild chunks")
    if parsed_args.report and results:
        write_report(results, parsed_args.report)
    return 0


def main() -> int:
    from rag.main import run_module

    return run_module(["ingest"])


if __name__ == "__main__":
    sys.exit(main())
