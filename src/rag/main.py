"""The `rag` command: one entry point for every stage of the pipeline.

    rag status                                   what is stored, is the index current, what is running, what next
    rag ingest arxiv corpus/seed_ai_research.txt --collection papers [--limit N] [--reparse] [--report out.csv]
    rag ingest files data/raw/manual/ --collection papers     # files or folders
    rag collections                              # collections with document and chunk counts
    rag ingest tei                               # re-apply TEI rules to saved GROBID output
    rag parse paper.pdf                          # debug view: one PDF through GROBID
    rag chunk [--show DOC_ID [--limit N]]
    rag index build | list
    rag search "query" [-k 5] [--mode hybrid|dense|bm25] [--rerank off|fused|replace]
               [--collection NAME ...] [--explain]
    rag ask "question" [--mode hybrid|dense|bm25] [--rerank off|fused|replace] [--collection NAME ...] [--explain]
    rag serve [--port 8000]                      # HTTP API on 127.0.0.1, docs at /docs
    rag sync NAME [--add-folder PATH] [--remove-folder PATH]   # keep a collection in step with its folders
    rag logs show [REQUEST_ID|last] [--json]     # one logged request as a trace
    rag logs stats [--last N] [--route ask|search]
    rag eval check|run --set papers   rag eval compare papers:latest~1 papers:latest   rag eval list

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
    for sub in (arxiv, files):
        sub.add_argument(
            "--collection", required=True, help="collection to add the documents to (created if new), e.g. papers"
        )
        sub.add_argument(
            "--parser",
            choices=["pymupdf", "grobid"],
            help="PDF parser (default: PDF_PARSER setting, pymupdf; grobid needs the GROBID server)",
        )
    for sub in (arxiv, files, tei):
        sub.add_argument("--reparse", action="store_true", help="parse again even if already parsed")
        sub.add_argument("--report", type=Path, help="write per-document results to this CSV")

    parse = command("parse", "rag.parse.__main__:run", "debug view: parse one PDF, print its sections, save JSON")
    parse.add_argument("pdf", type=Path)
    parse.add_argument("--parser", choices=["pymupdf", "grobid"], help="default: PDF_PARSER setting (pymupdf)")
    parse.add_argument("--doc-id", help='defaults to "file:<pdf name without .pdf>"')

    chunk = command("chunk", "rag.chunk.__main__:run", "chunk every parsed document (safe to rerun)")
    chunk.add_argument("--show", metavar="DOC_ID", help="print one document's chunks instead")
    chunk.add_argument("--limit", type=int, help="with --show: only the first N chunks")

    collections = command("collections", "rag.collections:run", "list collections with document and chunk counts")
    collections.add_argument("action", nargs="?", choices=["list"], default="list")

    evaluate = command("eval", "rag.eval.cli:run", "score retrieval on a question set, save and compare runs")
    actions = evaluate.add_subparsers(dest="action", required=True, metavar="ACTION")
    for name, help in (
        ("check", "check that every gold quote is found in exactly one chunk"),
        ("run", "run a question set through retrieval and save the run"),
    ):
        sub = actions.add_parser(name, help=help)
        sub.add_argument("--set", required=True, help="set name in eval/sets/ (papers, hr-demo) or a .jsonl path")
        if name == "run":
            sub.add_argument("--label", help="short name added to the run id, e.g. baseline")
            sub.add_argument("--no-filter", action="store_true", help="search all collections, not the question's")
            sub.add_argument(
                "--rerank", choices=["off", "fused", "replace"], help="reranker on top (default: RERANK setting)"
            )
            sub.add_argument(
                "--retriever",
                choices=["hybrid", "dense", "bm25"],
                help="default: RETRIEVAL_MODE setting (hybrid)",
            )
    compare = actions.add_parser("compare", help="compare two runs: metric deltas and questions that changed rank")
    compare.add_argument("a", help="run id, path, or SET:latest~N (e.g. papers:latest~1)")
    compare.add_argument("b", help="run id, path, or SET:latest")
    actions.add_parser("list", help="saved runs with headline metrics")
    answers = actions.add_parser("answers", help="run rag ask on every question of a set, with automatic checks")
    answers.add_argument("--set", required=True, help="set name in eval/sets/ or a .jsonl path")
    answers.add_argument("--runs", type=int, default=2, help="times each question is asked (default 2)")
    answers.add_argument("--retriever", choices=["hybrid", "dense", "bm25"], help="default: RETRIEVAL_MODE")
    answers.add_argument("--rerank", choices=["off", "fused", "replace"], help="default: ASK_RERANK (fused)")
    answers.add_argument("--prompt", choices=["v2", "v3"], help="answer prompt (default: ASK_PROMPT, v3)")
    answers.add_argument("--label", help="short name added to the run id")
    sheet = actions.add_parser("sheet", help="write a labeling sheet from answer runs")
    sheet.add_argument("runs", nargs="+", help="answer run ids, paths, or SET:latest")
    sheet.add_argument("--per-set", nargs="+", required=True, metavar="SET=N", help="entries per set, e.g. papers=11")
    sheet.add_argument("--out", required=True, help="where to write the sheet (.md)")
    gate = actions.add_parser("gate", help="quality gate: retrieval + guardrail replay vs eval/baseline.json")
    gate.add_argument("--update", metavar="REASON", help="record the current numbers as the new baseline")
    rescore = actions.add_parser("rescore", help="redo the automatic checks on a stored answer run (no LLM)")
    rescore.add_argument("run", help="answer run id, path, or SET:latest")
    labels = actions.add_parser("labels", help="compare a filled labeling sheet with the automatic checks")
    labels.add_argument("sheet", help="the filled .md sheet")

    index = command("index", "rag.index.__main__:run", "build or list dense indexes")
    index.add_argument("action", choices=["build", "list"])

    search = command("search", "rag.search:run", "dense search over the current index (no LLM)")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument(
        "--mode",
        choices=["hybrid", "dense", "bm25"],
        help="retriever (default: RETRIEVAL_MODE setting, hybrid)",
    )
    search.add_argument(
        "--collection",
        action="append",
        dest="collections",
        metavar="NAME",
        help="search only this collection (repeat for several; default: all)",
    )
    search.add_argument(
        "--rerank", choices=["off", "fused", "replace"], help="reranker on top (default: RERANK setting, off)"
    )
    search.add_argument("--explain", action="store_true", help="show timings, settings and chunk text")

    ask = command("ask", "rag.ask:run", "answer a question with checked citations")
    ask.add_argument("question")
    ask.add_argument(
        "--collection",
        action="append",
        dest="collections",
        metavar="NAME",
        help="search only this collection (repeat for several; default: all)",
    )
    ask.add_argument(
        "--mode",
        choices=["hybrid", "dense", "bm25"],
        help="retriever (default: RETRIEVAL_MODE setting, hybrid)",
    )
    ask.add_argument(
        "--rerank", choices=["off", "fused", "replace"], help="reranker on top (default: ASK_RERANK setting, fused)"
    )
    ask.add_argument("--explain", action="store_true", help="show retrieval, the exact prompt, tokens, timings")

    sync = command("sync", "rag.sync:run", "bring a collection in step with its folders and uploads")
    sync.add_argument("collection")
    sync.add_argument("--add-folder", type=Path, metavar="PATH", help="register this folder for the collection first")
    sync.add_argument(
        "--remove-folder", type=Path, metavar="PATH", help="forget this folder and delete its documents first"
    )

    logs = command("logs", "rag.requestlog:run", "per-request logs: one request as a trace, or stats per stage")
    log_actions = logs.add_subparsers(dest="action", required=True, metavar="ACTION")
    show = log_actions.add_parser("show", help="one request as a readable trace (default: the last one)")
    show.add_argument("request_id", nargs="?", help="request id or its start; 'last' (default)")
    show.add_argument("--json", action="store_true", help="print the raw record")
    log_stats = log_actions.add_parser("stats", help="p50 / p95 per stage, error / fallback / refusal counts")
    log_stats.add_argument("--last", type=int, default=200, help="only the last N requests (default 200)")
    log_stats.add_argument("--route", choices=["ask", "search"])
    log_stats.add_argument("--json", action="store_true")

    serve = command("serve", "rag.api.app:run", "run the HTTP API (search, ask with streaming, health)")
    serve.add_argument("--host", help="default: API_HOST setting (127.0.0.1); only loopback is allowed")
    serve.add_argument("--port", type=int, help="default: API_PORT setting (8000)")
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
