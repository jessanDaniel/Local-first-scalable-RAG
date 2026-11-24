"""Ingest documents: source -> PDF -> parse -> SQLite.

    rag ingest arxiv corpus/seed_ai_research.txt [--limit N]
    rag ingest files data/raw/manual/            # or individual .pdf paths
    rag ingest tei      # re-apply TEI rules to saved GROBID output, no GROBID

Two sources so far. `arxiv`: metadata from the arXiv API (title, authors,
abstract, dates come from there, not from the PDF), PDF downloaded to
data/raw/arxiv/. `files`: local PDFs, copied to data/raw/files/; title and
authors come from the parse, or the file name when the parse has none.

Every document then goes the same way: the PDF parser (PDF_PARSER / --parser: "pymupdf", the
layout parser in rag/parse/pdf.py, by default; "grobid" for papers, falling back to pymupdf on a
parse error), or the text parser for .txt/.md; document + sections written to the database as
ok / suspect / failed.
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
from rag.models import DEFAULT_COLLECTION, Document, check_collection_id, doc_file_stem, make_doc_id
from rag.parse.base import ParsedDocument, ParseError, Parser, ParserUnavailableError
from rag.parse.grobid import GrobidParser
from rag.parse.office import DocxParser, HTMLDocParser
from rag.parse.pdf import LayoutPDFParser
from rag.parse.pymupdf import pdf_title_candidates
from rag.parse.tei import tei_to_parsed
from rag.parse.text import TEXT_SUFFIXES, TextParser
from rag.parse.title import choose_title
from rag.store import db

# The PDF parser reads the body; this issue is covered by the source's metadata or the file name.
METADATA_ISSUES = {"no title"}


@dataclass
class FoundFile:
    path: Path
    key: str  # what the doc_id is made from: the file name, or the path inside the folder given


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
    fallback: Parser | None,
    parsed_dir: Path,
    downloaded: bool = False,
    download_seconds: float = 0.0,
) -> IngestResult:
    start = time.perf_counter()
    used_fallback = False
    errors: list[str] = []
    parsed: ParsedDocument | None = None
    for parser in [p for p in (primary, fallback) if p is not None]:
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

    document, status, issues = store_parsed(conn, document, parsed, errors, parsed_dir, pdf_path)
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
    source_path: Path,
) -> tuple[Document, str, list[str]]:
    """Write a parse result: document row, its sections (old chunks go with them), JSON dump."""
    issues = [i for i in parsed.issues if i not in METADATA_ISSUES] + errors
    metadata = dict(document.metadata)
    if document.source == "file":
        # No source metadata: the title is chosen from the parse, the PDF itself, then the file name.
        original = Path(metadata["original_path"]) if metadata.get("original_path") else source_path
        title, title_source, rejected = pick_title(parsed, source_path, original.stem)
        metadata.update(title_source=title_source, title_rejected=rejected)
        if title_source != "parser":
            issues.append(f"title guessed from {title_source}")
    else:
        title = document.title or parsed.title or source_path.stem
    status = "suspect" if issues else "ok"
    document = replace(
        document,
        # Source metadata wins; the parse fills only what the source does not know.
        title=title,
        authors=document.authors or parsed.authors,
        abstract=document.abstract or parsed.abstract,
        metadata=metadata,
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


def pick_title(parsed: ParsedDocument, source_path: Path, file_name: str | None = None) -> tuple[str, str, list[str]]:
    """file_name: the name the user gave the file (not our stored copy's "file_<id>")."""
    candidates = [("parser", parsed.title)]
    if source_path.suffix.lower() == ".pdf":
        from_pdf = pdf_title_candidates(source_path)
        candidates += [("pdf metadata", from_pdf["pdf metadata"]), ("largest font", from_pdf["largest font"])]
    candidates.append(("file name", file_name or source_path.stem))
    return choose_title(candidates)


class Ingestor:
    """What every source shares: the database, the parsers, the output folders."""

    def __init__(
        self,
        settings: Settings,
        reparse: bool = False,
        collection: str = DEFAULT_COLLECTION,
        pdf_parser: str | None = None,
    ):
        self.settings = settings
        self.pdf_parser = pdf_parser or settings.pdf_parser
        self.reparse = reparse
        self.collection = check_collection_id(collection)
        self.parsed_dir = settings.data_dir / "parsed"
        self.conn = db.open_db(settings.db_path)
        self.grobid = GrobidParser.from_settings(settings, tei_dir=self.parsed_dir)
        self.layout = LayoutPDFParser()
        self.text_parser = TextParser()
        self.docx_parser = DocxParser()
        self.html_parser = HTMLDocParser()
        self._checked_grobid = False

    def close(self) -> None:
        self.conn.close()

    def already_parsed(self, doc_id: str, sha256: str | None = None) -> Document | None:
        """The stored document when it can be skipped, else None."""
        existing = db.get_document(self.conn, doc_id)
        if existing is not None and existing.collection_id != self.collection:
            # One collection per document: moving one is a deliberate act, never a side effect.
            raise ValueError(f"{doc_id} is already in collection {existing.collection_id!r}, not {self.collection!r}")
        if existing is None or existing.parse_status == "pending" or self.reparse:
            return None
        if sha256 is not None and existing.source_sha256 != sha256:
            return None  # the file changed since it was parsed
        return existing

    def parse(self, document: Document, pdf_path: Path, **timing) -> IngestResult:
        suffix = pdf_path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            primary, fallback = self.text_parser, None  # no service needed, nothing to fall back to
        elif suffix == ".docx":
            primary, fallback = self.docx_parser, None
        elif suffix in HTML_SUFFIXES:
            primary, fallback = self.html_parser, None
        elif self.pdf_parser == "grobid":
            primary, fallback = self.grobid, self.layout
            self._require_grobid()
        else:
            primary, fallback = self.layout, None
        db.upsert_document(self.conn, replace(document, title=document.title or pdf_path.stem))
        result = parse_and_store(document, pdf_path, self.conn, primary, fallback, self.parsed_dir, **timing)
        print(
            f"{result.status:8} {result.doc_id:26} {result.parse_seconds:6.2f}s "
            f"{result.n_sections:3} sections {result.body_words:6} words  {result.title[:50]}",
            flush=True,
        )
        return result

    def _require_grobid(self) -> None:
        if not self._checked_grobid:
            if not self.grobid.is_alive():
                raise ParserUnavailableError(
                    f"GROBID is not running at {self.grobid.base_url}; start it (see AGENTS.md) "
                    "or use the default parser (--parser pymupdf)"
                )
            self._checked_grobid = True

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
            document = replace(document, source_sha256=sha256, collection_id=self.collection)
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
            source_path = Path(document.source_uri) if document.source == "file" else tei_path
            document, status, issues = store_parsed(self.conn, document, parsed, [], self.parsed_dir, source_path)
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

    def ingest_files(self, paths: list[Path] | list[FoundFile]) -> list[IngestResult]:
        files_dir = self.settings.data_dir / "raw" / "files"
        results = []
        for item in paths:
            path, key = (item.path, item.key) if isinstance(item, FoundFile) else (item, item.stem)
            source_format = file_format(path)
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            doc_id = make_doc_id("file", key)
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
            copy = files_dir / f"{doc_file_stem(doc_id)}.{source_format}"  # .htm is stored as .html
            shutil.copyfile(path, copy)
            document = Document(
                doc_id=doc_id,
                source="file",
                title="",
                source_uri=str(copy),
                source_format=source_format,
                source_sha256=sha256,
                metadata={"original_path": original},
                collection_id=self.collection,
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


HTML_SUFFIXES = {".html": "html", ".htm": "html"}
FORMATS = {".pdf": "pdf", ".docx": "docx", **TEXT_SUFFIXES, **HTML_SUFFIXES}
MAX_FILE_BYTES = 200_000_000


def file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in FORMATS:
        return FORMATS[suffix]
    raise ValueError(f"{path}: unsupported file type (supported: {', '.join(sorted(FORMATS))})")


def document_paths(inputs: list[Path]) -> list[Path]:
    """The files to ingest (see find_documents); skipped files are not listed."""
    return [f.path for f in find_documents(inputs)[0]]


def find_documents(inputs: list[Path]) -> tuple[list[FoundFile], list[tuple[Path, str]]]:
    """Files as given; folders are walked recursively. Returns (files, skipped) where skipped holds
    every file left out and why: hidden, unsupported type, empty, too large, or the same content as
    a file already found. Symlinked folders are not followed (no loops). A file inside a folder is
    keyed by its path in that folder ("policies/travel.pdf" -> "policies-travel"), so two files with
    the same name in different subfolders get different documents."""
    import os

    found: list[FoundFile] = []
    skipped: list[tuple[Path, str]] = []
    seen: dict[str, Path] = {}

    def consider(path: Path, key: str) -> None:
        if path.suffix.lower() not in FORMATS:
            skipped.append((path, f"unsupported type {path.suffix or '(none)'}"))
            return
        size = path.stat().st_size
        if size == 0:
            skipped.append((path, "empty"))
            return
        if size > MAX_FILE_BYTES:
            skipped.append((path, f"larger than {MAX_FILE_BYTES // 1_000_000} MB"))
            return
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            skipped.append((path, f"same content as {seen[digest]}"))
            return
        seen[digest] = path
        found.append(FoundFile(path, key))

    for item in inputs:
        if not item.is_dir():
            consider(item, item.stem)
            continue
        for folder, dirs, files in os.walk(item, followlinks=False):
            hidden = [d for d in dirs if d.startswith(".")]
            for d in hidden:
                skipped.append((Path(folder) / d, "hidden folder"))
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                path = Path(folder) / name
                if name.startswith(".") or name.startswith("~$"):  # hidden files, Word lock files
                    skipped.append((path, "hidden or temporary file"))
                    continue
                relative = path.relative_to(item).with_suffix("")
                consider(path, "-".join(relative.parts))
    return found, skipped


def run(parsed_args) -> int:
    collection = getattr(parsed_args, "collection", None)  # tei re-applies rules in place: no collection
    ingestor = Ingestor(
        get_settings(),
        reparse=parsed_args.reparse,
        collection=collection or DEFAULT_COLLECTION,
        pdf_parser=getattr(parsed_args, "parser", None),
    )
    if collection and db.ensure_collection(ingestor.conn, collection):
        print(f"created collection {collection!r}")
    try:
        if parsed_args.source == "arxiv":
            results = ingestor.ingest_arxiv(read_id_file(parsed_args.id_file)[: parsed_args.limit])
        elif parsed_args.source == "files":
            found, skipped_files = find_documents(parsed_args.paths)
            for path, why in skipped_files:
                print(f"skipped  {path}  ({why})")
            results = ingestor.ingest_files(found)
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
