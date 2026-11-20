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
    from rag.eval.runner import git_commit, score_questions
    from rag.eval.sets import file_sha1
    from rag.index.current import StaleIndexError
    from rag.search import make_searcher

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
        searcher = make_searcher(conn, args.retriever or settings.retrieval_mode)
    except StaleIndexError as exc:
        print(exc)
        return 1
    first_gold = {qid: ids[0] for qid, ids in gold.items() if ids}
    texts = {c.chunk_id: c.text for c in db.get_chunks_by_ids(conn, list(first_gold.values()))}
    gold_texts = {qid: texts[cid] for qid, cid in first_gold.items()}
    searcher.search("warm-up query, not scored", 5)  # load models and caches before anything is timed
    rows = score_questions(searcher, questions, gold, not args.no_filter, gold_texts)

    config = {"retriever": searcher.mode, **retriever_config(searcher, conn)}
    config |= {
        "collection_filter": not args.no_filter,
        "k": max(len(r["ranked"]) for r in rows),
        "set_file": path.name,
        "set_sha1": file_sha1(path),
        "git_commit": git_commit(),
    }
    from rag.eval.runner import new_run, print_run, save_run

    run = new_run(path.stem, config, rows, args.label)
    saved = save_run(run, runs_dir())
    print_run(run)
    print(f"\nsaved {saved}")
    return 0


def retriever_config(searcher, conn) -> dict:
    """What a run must record to be reproduced: the index and every setting that shapes the ranking."""
    if searcher.mode == "rerank":
        r = searcher.reranker
        return {
            **retriever_config(searcher.hybrid, conn),
            "index_type": f"{retriever_config(searcher.hybrid, conn)['index_type']} + cross-encoder",
            "rerank_model": r.model_name,
            "rerank_device": r.device,
            "rerank_candidates": searcher.candidates,
            "rerank_keep": searcher.keep,
        }
    if searcher.mode == "hybrid":
        dense = retriever_config(searcher.dense, conn)
        bm25 = retriever_config(searcher.bm25, conn)
        return {
            **dense,
            **{k: v for k, v in bm25.items() if k.startswith(("bm25", "tokenizer"))},
            "index_type": f"{dense['index_type']} + BM25, RRF",
            "rrf_k": searcher.rrf_k,
            "hybrid_candidates": searcher.candidates,
        }
    if searcher.mode == "dense":
        m = searcher.index.manifest
        return {
            "index_build": m.build_id,
            "index_type": m.index_type,
            "n_chunks": m.n_chunks,
            "embedding_model": m.embedding_model,
            "query_instruction": searcher.embedder.query_instruction,
            "chunker_version": m.chunker_version,
        }
    from rag.index.tokenize import TOKENIZER_VERSION

    index = searcher.index
    return {
        "index_build": "in-memory",
        "index_type": "BM25 (hand-written)",
        "n_chunks": index.n_chunks,
        "bm25_k1": index.k1,
        "bm25_b": index.b,
        "tokenizer": TOKENIZER_VERSION,
        "chunker_version": ", ".join(db.stats(conn)["chunks_by_chunker"]),
    }


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
