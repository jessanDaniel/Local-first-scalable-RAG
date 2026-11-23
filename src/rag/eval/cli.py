"""`rag eval check | run | compare | list`.

rag eval check --set papers            every quote found in exactly one chunk?
rag eval run --set papers [--label baseline] [--no-filter]
rag eval compare latest~1 latest       metric deltas + questions whose rank changed
rag eval list
"""

import sys
from pathlib import Path

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
        searcher = make_searcher(conn, args.retriever or settings.retrieval_mode, args.rerank or settings.rerank)
    except StaleIndexError as exc:
        print(exc)
        return 1
    first_gold = {qid: ids[0] for qid, ids in gold.items() if ids}
    texts = {c.chunk_id: c.text for c in db.get_chunks_by_ids(conn, list(first_gold.values()))}
    gold_texts = {qid: texts[cid] for qid, cid in first_gold.items()}
    searcher.search("warm-up query, not scored", 5)  # load models and caches before anything is timed
    rows = score_questions(searcher, questions, gold, not args.no_filter, gold_texts)

    base = searcher.base if searcher.mode == "rerank" else searcher
    rerank = searcher.strategy if searcher.mode == "rerank" else "off"
    config = {"retriever": base.mode, "rerank": rerank, **retriever_config(searcher, conn)}
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
        base = retriever_config(searcher.base, conn)
        return {
            **base,
            "index_type": f"{base['index_type']} + cross-encoder ({searcher.strategy})",
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


def answers_dir():
    return get_settings().data_dir / "eval" / "answers"


def answers(args) -> int:
    """Run `rag ask` on every question of a set, `--runs` times, with automatic checks."""
    import hashlib

    from rag.answer.answerer import Answerer
    from rag.chunk.tokens import HFTokenCounter
    from rag.eval.answers import print_answer_summary, run_answers, save_answer_run
    from rag.eval.runner import git_commit
    from rag.eval.sets import file_sha1
    from rag.index.current import StaleIndexError
    from rag.llm.factory import create_llm
    from rag.search import make_searcher
    from rag.status import probe_ollama

    settings = get_settings()
    if settings.llm_provider == "ollama":
        state = probe_ollama(settings.ollama_base_url, settings.ollama_model)
        if state != "running":
            print(f"Ollama: {state}. Start it first (brew services start ollama); nothing was run.")
            return 1
    path = set_path(args.set)
    questions = load_set(path)
    conn = db.open_db(settings.db_path)
    gold, checks = resolve_gold(conn, questions)
    if any(c.status in ("split", "missing") for c in checks):
        print(f"some quotes have no chunk: run `rag eval check --set {args.set}` first")
        return 1
    retriever = args.retriever or settings.retrieval_mode
    rerank = args.rerank or settings.ask_rerank
    try:
        searcher = make_searcher(conn, retriever, rerank)
    except StaleIndexError as exc:
        print(exc)
        return 1
    llm = create_llm(settings)
    answerer = Answerer(
        conn,
        searcher,
        llm,
        HFTokenCounter(settings.llm_tokenizer),
        settings.ask_retrieve_k,
        settings.ask_max_sources,
        settings.ask_context_tokens,
        prompt_version=args.prompt or settings.ask_prompt,
        guardrails=settings.ask_guardrails,
        sanitize_sources=settings.ask_sanitize_sources,
    )
    print("warming up (models load once) ...", flush=True)
    answerer.ask("warm-up question, not scored", [questions[0].collection])
    records = run_answers(answerer, questions, gold, args.runs, progress=lambda line: print(line, flush=True))
    config = {
        "llm": f"{llm.provider} {llm.model}",
        "temperature": settings.llm_temperature,
        "context_window": llm.context_window,
        "max_answer_tokens": llm.max_tokens,
        "system_prompt_sha1": hashlib.sha1(answerer.system_prompt.encode()).hexdigest()[:12],
        "retriever": retriever,
        "rerank": rerank,
        "prompt": answerer.prompt_version,
        "guardrails": answerer.guardrails,
        "sanitize_sources": answerer.sanitize_sources,
        "rerank_model": settings.rerank_model if rerank != "off" else None,
        "retrieve_k": settings.ask_retrieve_k,
        "max_sources": settings.ask_max_sources,
        "context_tokens": settings.ask_context_tokens,
        "runs": args.runs,
        "set_file": path.name,
        "set_sha1": file_sha1(path),
        "git_commit": git_commit(),
    }
    run = save_answer_run(path.stem, config, records, answers_dir(), args.label)
    print()
    print_answer_summary(run)
    print(f"\nsaved {answers_dir() / (run['run_id'] + '.json')}")
    return 0


def sheet(args) -> int:
    """Write a labeling sheet: a varied sample of answers from one or more answer runs."""
    from rag.eval.labels import write_sheet
    from rag.eval.runner import load_run

    runs = [load_run(ref, answers_dir()) for ref in args.runs]
    per_run = {}
    for spec in args.per_set:
        name, _, count = spec.partition("=")
        per_run[name] = int(count)
    out = Path(args.out)
    n = write_sheet(runs, per_run, out)
    print(f"wrote {n} entries to {out}")
    return 0


def gate(args) -> int:
    """Quality gate: retrieval and guardrail replay vs eval/baseline.json, zero tolerance."""
    import time

    from rag.eval import gate as g
    from rag.eval.runner import git_commit
    from rag.index.current import StaleIndexError
    from rag.search import make_searcher

    start = time.perf_counter()
    settings = get_settings()
    if not settings.db_path.exists():
        print("quality gate skipped: no database on this machine (data/ is not in git)")
        return 0
    conn = db.open_db(settings.db_path)
    try:
        searcher = make_searcher(conn, settings.retrieval_mode, settings.rerank)
    except StaleIndexError as exc:
        print(f"quality gate FAILED: {exc}")
        return 1
    retrieval = g.retrieval_snapshot(conn, searcher)
    flags = g.replay_flags(conn, answers_dir())
    base = searcher.base if searcher.mode == "rerank" else searcher
    config = {
        "retriever": base.mode,
        "rerank": getattr(searcher, "strategy", "off"),
        **{k: v for k, v in retriever_config(searcher, conn).items() if k != "index_build"},
    }
    previous = g.load_baseline()

    if args.update:
        baseline = g.write_baseline(retrieval, flags, config, args.update, git_commit(), previous)
        for name, snap in baseline["retrieval"].items():
            m = snap["metrics"]
            print(
                f"{name:10} hit@1 {m['hit@1']}/{snap['n']}  hit@5 {m['hit@5']}/{snap['n']}  "
                f"hit@10 {m['hit@10']}/{snap['n']}  mrr {m['mrr']:.3f}"
            )
        print(f"guardrail replay: {len(baseline['flags'] or {})} stored answers")
        print(f"baseline recorded in {g.BASELINE_PATH.relative_to(g.PROJECT_ROOT)}: {args.update}")
        return 0
    if previous is None:
        print('quality gate FAILED: no baseline yet; record one with rag eval gate --update "initial baseline"')
        return 1

    failures, improvements = g.compare_retrieval(previous["retrieval"], retrieval)
    if flags is None:
        print("guardrail replay skipped: the stored answer runs are not on this machine")
    elif previous.get("flags") is not None:
        failures += g.compare_flags(previous["flags"], flags)
    changed = {
        k for k in set(config) | set(previous.get("config", {})) if config.get(k) != previous.get("config", {}).get(k)
    }
    if changed:
        print(
            "note: settings differ from the baseline's: "
            + ", ".join(f"{k} {previous['config'].get(k)!r} -> {config.get(k)!r}" for k in sorted(changed))
        )
    for line in improvements:
        print(f"better  {line}")
    for line in failures:
        print(f"WORSE   {line}")
    seconds = time.perf_counter() - start
    if failures:
        print(
            f"quality gate FAILED ({len(failures)} regressions, {seconds:.0f} s). If intended, record it: "
            'rag eval gate --update "why"'
        )
        return 1
    note = f"; {len(improvements)} improvements: record them with --update" if improvements else ""
    print(f"quality gate passed: {', '.join(g.GATE_SETS)} retrieval + guardrail replay ({seconds:.0f} s){note}")
    return 0


def rescore_answers(args) -> int:
    """Redo the automatic checks on a stored answer run (after a check was improved)."""
    import json

    from rag.eval.answers import print_answer_summary, rescore
    from rag.eval.runner import load_run

    run = rescore(load_run(args.run, answers_dir()))
    (answers_dir() / f"{run['run_id']}.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    print_answer_summary(run)
    return 0


def labels(args) -> int:
    """Read a filled sheet and report how well the automatic checks agree with the labels."""
    import json

    from rag.eval.labels import agreement_report, parse_sheet

    found, problems = parse_sheet(Path(args.sheet).read_text())
    for problem in problems:
        print(f"PROBLEM: {problem}")
    runs = {}
    for label in found:
        if label["run_id"] not in runs:
            runs[label["run_id"]] = json.loads((answers_dir() / f"{label['run_id']}.json").read_text())
    print("\n".join(agreement_report(found, runs)))
    return 1 if problems else 0


def run(args) -> int:
    actions = {
        "check": check,
        "run": run_eval,
        "compare": compare,
        "list": list_all,
        "answers": answers,
        "sheet": sheet,
        "labels": labels,
        "rescore": rescore_answers,
        "gate": gate,
    }
    try:
        return actions[args.action](args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    from rag.main import run_module

    sys.exit(run_module(["eval"]))
