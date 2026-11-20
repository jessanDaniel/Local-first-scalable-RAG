"""`rag eval check | run | compare | list`.

rag eval check --set papers            every quote found in exactly one chunk?
rag eval run --set papers [--label baseline] [--no-filter]
rag eval compare latest~1 latest       metric deltas + questions whose rank changed
rag eval list
"""

import sys

from rag.config import get_settings
from rag.eval.sets import load_set, resolve_gold, set_path
from rag.store import db


def runs_dir():
    return get_settings().data_dir / "eval" / "runs"


def check(args) -> int:
    settings = get_settings()
    path = set_path(args.set)
    questions = load_set(path)
    conn = db.open_db(settings.db_path)
    gold, checks = resolve_gold(conn, questions)
    bad = [c for c in checks if c.status != "one chunk"]
    answerable = sum(q.answerable for q in questions)
    print(f"{path.name}: {len(questions)} questions ({answerable} answerable), {len(checks)} quotes")
    counts = {
        status: sum(c.status == status for c in checks)
        for status in ("one chunk", "several chunks", "split", "missing")
    }
    print("quotes: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for c in bad:
        print(f"  {c.status:14} {c.question_id}: {c.quote[:80]!r}")
    missing_docs = sorted({q.doc_id for q in questions if q.answerable and not gold.get(q.id)})
    if missing_docs:
        print(f"no gold for documents: {', '.join(missing_docs)} (not ingested or chunked?)")
    return 1 if any(c.status in ("split", "missing") for c in checks) else 0


def run_eval(args) -> int:
    from rag.embed.embedder import Embedder
    from rag.eval.runner import git_commit, new_run, print_run, save_run, score_questions
    from rag.index.dense import StaleIndexError, load_current
    from rag.search import DenseSearcher

    settings = get_settings()
    path = set_path(args.set)
    questions = load_set(path)
    conn = db.open_db(settings.db_path)
    gold, checks = resolve_gold(conn, questions)
    broken = [c for c in checks if c.status in ("split", "missing")]
    if broken:
        print(f"{len(broken)} quotes have no chunk: run `rag eval check --set {args.set}` first")
        return 1
    try:
        index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model, conn)
    except StaleIndexError as exc:
        print(exc)
        return 1
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)
    searcher = DenseSearcher(conn, index, embedder)
    first_gold = {qid: ids[0] for qid, ids in gold.items() if ids}
    texts = {c.chunk_id: c.text for c in db.get_chunks_by_ids(conn, list(first_gold.values()))}
    gold_texts = {qid: texts[cid] for qid, cid in first_gold.items()}
    rows = score_questions(searcher, questions, gold, not args.no_filter, gold_texts)
    m = index.manifest
    from rag.eval.sets import file_sha1

    config = {
        "retriever": "dense",
        "index_build": m.build_id,
        "index_type": m.index_type,
        "n_chunks": m.n_chunks,
        "embedding_model": m.embedding_model,
        "query_instruction": settings.embedding_query_instruction,
        "chunker_version": m.chunker_version,
        "collection_filter": not args.no_filter,
        "k": max(len(r["ranked"]) for r in rows),
        "set_file": path.name,
        "set_sha1": file_sha1(path),
        "git_commit": git_commit(),
    }
    run = new_run(path.stem, config, rows, args.label)
    saved = save_run(run, runs_dir())
    print_run(run)
    print(f"\nsaved {saved}")
    return 0


def compare(args) -> int:
    from rag.eval.runner import compare_runs, load_run

    try:
        a, b = load_run(args.a, runs_dir()), load_run(args.b, runs_dir())
        lines = compare_runs(a, b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print("\n".join(lines))
    return 0


def list_all(args) -> int:
    import json

    from rag.eval.runner import list_runs

    for path in list_runs(runs_dir()):
        run = json.loads(path.read_text())
        o = run["summary"]["overall"]
        print(f"{run['run_id']:45} hit@1 {o['hit@1']:>3}/{o['n']}  hit@5 {o['hit@5']:>3}/{o['n']}  mrr {o['mrr']:.3f}")
    return 0


def run(args) -> int:
    actions = {"check": check, "run": run_eval, "compare": compare, "list": list_all}
    try:
        return actions[args.action](args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    from rag.main import run_module

    sys.exit(run_module(["eval"]))
