"""The `rag` command: one entry point for every stage of the pipeline.

    rag status                                   what is stored, is the index current, what is running, what next
    rag ingest arxiv corpus/seed_ai_research.txt [--limit N] [--reparse] [--report out.csv]
    rag ingest files data/raw/manual/            # PDFs or folders of PDFs
    rag ingest tei                               # re-apply TEI rules to saved GROBID output
    rag parse paper.pdf                          # debug view: one PDF through GROBID
    rag chunk [--show DOC_ID [--limit N]]
    rag index build | list
    rag search "query" [-k 5] [--explain]
    rag ask "question" [--explain]

`--data-dir DIR` (before the command) points everything at another data folder,
e.g. a scratch copy. The `python -m rag.<module>` entry points still work and
call this same code.

Argument definitions live here; each command's code is imported only when that
command runs, so `rag --help` and `rag status` never load torch or FAISS.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

DEFAULT_SEED = "corpus/seed_ai_research.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Production RAG pipeline: ingest, chunk, index, search, ask."
    )
    parser.add_argument("--data-dir", type=Path, help="use this data folder instead of the configured one")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, handler: str, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help, description=help)
        sub.set_defaults(handler=handler)
        return sub

    status = command("status", "rag.status:run", "show stored documents, index state, services and the next step")
    status.add_argument("--json", action="store_true", help="print the report as JSON")

    ingest = command("ingest", "rag.ingest:run", "fetch and parse documents into the database")
    sources = ingest.add_subparsers(dest="source", required=True, metavar="SOURCE")
    arxiv = sources.add_parser("arxiv", help="arXiv papers listed in an id file")
    arxiv.add_argument("id_file", type=Path, help="one arXiv id per line, '#' starts a comment")
    arxiv.add_argument("--limit", type=int, help="only the first N ids")
    files = sources.add_parser("files", help="local PDF files or folders of PDFs")
    files.add_argument("paths", type=Path, nargs="+")
    tei = sources.add_parser("tei", help="re-apply the TEI rules to saved GROBID output (no GROBID needed)")
    for sub in (arxiv, files, tei):
        sub.add_argument("--reparse", action="store_true", help="parse again even if already parsed")
        sub.add_argument("--report", type=Path, help="write per-document results to this CSV")

    parse = command("parse", "rag.parse.__main__:run", "debug view: parse one PDF with GROBID, save TEI + JSON")
    parse.add_argument("pdf", type=Path)
    parse.add_argument("--doc-id", help='defaults to "file:<pdf name without .pdf>"')

    chunk = command("chunk", "rag.chunk.__main__:run", "chunk every parsed document (safe to rerun)")
    chunk.add_argument("--show", metavar="DOC_ID", help="print one document's chunks instead")
    chunk.add_argument("--limit", type=int, help="with --show: only the first N chunks")

    index = command("index", "rag.index.__main__:run", "build or list dense indexes")
    index.add_argument("action", choices=["build", "list"])

    search = command("search", "rag.search:run", "dense search over the current index (no LLM)")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--explain", action="store_true", help="show timings, settings and chunk text")

    ask = command("ask", "rag.ask:run", "answer a question with checked citations")
    ask.add_argument("question")
    ask.add_argument("--explain", action="store_true", help="show retrieval, the exact prompt, tokens, timings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_dir is not None:
        # Settings read DATA_DIR from the environment; set it before anything loads them.
        os.environ["DATA_DIR"] = str(args.data_dir.resolve())
        from rag.config import get_settings

        get_settings.cache_clear()
    module_name, function_name = args.handler.split(":")
    handler = getattr(importlib.import_module(module_name), function_name)
    return handler(args)


def run_module(command: list[str]) -> int:
    """For the `python -m rag.<module>` entry points: same parser, same code."""
    return main([*command, *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
