"""Read-only views of stored evaluation results, for the web UI (step 22).

    retrieval runs   data/eval/runs/*.json      (rag eval run)
    answer runs      data/eval/answers/*.json   (rag eval answers)
    gate baseline    eval/baseline.json         (rag eval gate --update)

Nothing here runs an evaluation or writes a file: running stays with the CLI, so every number in
the UI comes from a run someone made on purpose, with its configuration saved next to it.

Run ids from a request are looked up in the directory listing (`{file stem: path}`), never joined
into a path, so "../" or an absolute path can't name a file outside the runs directory.

Files are parsed once and cached by (modification time, size): a run file never changes after it
is written (rescore rewrites answer runs, which changes the mtime).
"""

import json
import threading
from pathlib import Path

from rag.eval.answers import summarise_answers
from rag.eval.runner import retriever_and_rerank, run_brief, run_diff
from rag.eval.sets import load_set, set_path

_cache: dict[Path, tuple[int, int, dict]] = {}
_lock = threading.Lock()


def read_json(path: Path) -> dict:
    stat = path.stat()
    with _lock:
        hit = _cache.get(path)
        if hit and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
            return hit[2]
    data = json.loads(path.read_text(encoding="utf-8"))
    with _lock:
        _cache[path] = (stat.st_mtime_ns, stat.st_size, data)
    return data


def files_by_id(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.glob("*.json")} if directory.is_dir() else {}


def run_by_id(directory: Path, run_id: str) -> dict | None:
    path = files_by_id(directory).get(run_id)
    return read_json(path) if path else None


# --- retrieval runs ------------------------------------------------------------------------------


def list_retrieval_runs(runs_dir: Path, set_name: str | None = None) -> list[dict]:
    """Newest first. A file that isn't a readable run is skipped, not an error."""
    out = []
    for path in files_by_id(runs_dir).values():
        try:
            run = read_json(path)
            if set_name is None or run["set"] == set_name:
                out.append(run_brief(run))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(out, key=lambda r: (r["created"], r["run_id"]), reverse=True)


def question_texts(set_name: str) -> dict[str, str]:
    """id -> question, from the set file as it is now (the run stores ids, not the wording)."""
    try:
        return {q.id: q.question for q in load_set(set_path(set_name))}
    except (OSError, ValueError):
        return {}


def retrieval_run_detail(runs_dir: Path, run_id: str) -> dict | None:
    run = run_by_id(runs_dir, run_id)
    if run is None:
        return None
    texts = question_texts(run["set"])
    retriever, rerank = retriever_and_rerank(run["config"])
    questions = [
        {
            "id": q["id"],
            "type": q["type"],
            "tags": q.get("tags", []),
            "collection": q.get("collection"),
            "answerable": q["answerable"],
            "question": texts.get(q["id"]),
            "first_rank": q.get("first_rank"),
            "n_gold": len(q.get("gold") or []),
            "overlap": q.get("overlap"),
            "top1_doc": q.get("top1_doc"),
            "top1_score": q.get("top1_score"),
            "latency_ms": q.get("latency_ms"),
            "fallback": q.get("fallback"),
        }
        for q in run["questions"]
    ]
    return {
        **run_brief(run),
        "config": run["config"],
        "retriever": retriever,
        "rerank": rerank,
        "summary": run["summary"],
        "questions": questions,
        "question_text_from": f"eval/sets/{run['set']}.jsonl (as it is now)" if texts else None,
    }


def compare(runs_dir: Path, a_id: str, b_id: str) -> dict:
    """run_diff of two stored runs. KeyError: unknown id. ValueError: different sets."""
    a, b = run_by_id(runs_dir, a_id), run_by_id(runs_dir, b_id)
    missing = [rid for rid, run in ((a_id, a), (b_id, b)) if run is None]
    if missing:
        raise KeyError(f"no run {', '.join(missing)}")
    diff = run_diff(a, b)
    texts = question_texts(a["set"])
    for flip in diff["flips"]:
        flip["question"] = texts.get(flip["id"])
    return diff


# --- answer runs ---------------------------------------------------------------------------------


def list_answer_runs(answers_dir: Path) -> list[dict]:
    """Newest first: what was run and the automatic summary (no per-answer text)."""
    out = []
    for path in files_by_id(answers_dir).values():
        try:
            run = read_json(path)
            c = run["config"]
            out.append(
                {
                    "run_id": run["run_id"],
                    "set": run["set"],
                    "created": run["created"],
                    "rescored": run.get("rescored"),
                    "llm": c.get("llm"),
                    "prompt": c.get("prompt", "v2"),
                    "guardrails": c.get("guardrails", False),
                    "sanitize_sources": c.get("sanitize_sources"),
                    "retriever": c.get("retriever"),
                    "rerank": c.get("rerank"),
                    "runs": c.get("runs"),
                    "git_commit": c.get("git_commit"),
                    "summary": answer_summary(run),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(out, key=lambda r: (r["created"], r["run_id"]), reverse=True)


def answer_summary(run: dict) -> dict:
    """Runs saved before a summary field existed get it recomputed from their records, when the
    records have what it needs; otherwise the stored summary as it is (the UI shows "—")."""
    if "injections" in run["summary"]:
        return run["summary"]
    try:
        return summarise_answers(run["questions"])
    except (KeyError, TypeError):
        return run["summary"]


# --- quality gate --------------------------------------------------------------------------------


def gate_summary(baseline_path: Path) -> dict | None:
    """The committed baseline without its per-question ranks and per-answer flags (counts instead)."""
    if not baseline_path.exists():
        return None
    b = read_json(baseline_path)
    flags = b.get("flags") or {}
    return {
        "updated": b.get("updated"),
        "commit": b.get("commit"),
        "reason": b.get("reason"),
        "config": b.get("config"),
        "sets": {
            name: {"n": s.get("n"), "metrics": s.get("metrics"), "misses_top5": misses(s.get("ranks") or {})}
            for name, s in (b.get("retrieval") or {}).items()
        },
        "answers_replayed": len(flags),
        "answers_flagged": sum(bool(v) for v in flags.values()),
        "history": list(reversed(b.get("history") or [])),
    }


def misses(ranks: dict) -> list[dict]:
    """Questions whose gold is not in the baseline's top 5 (what the gate allows to stay missed)."""
    return sorted(
        ({"id": qid, "rank": rank} for qid, rank in ranks.items() if rank is None or rank > 5),
        key=lambda m: (m["rank"] is None, m["rank"] or 0, m["id"]),
    )
