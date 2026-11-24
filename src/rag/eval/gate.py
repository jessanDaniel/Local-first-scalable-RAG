"""The quality gate: is the system still as good as the recorded baseline? Zero tolerance.

Two checks, both deterministic and LLM-free, so any change is a real effect of the code:

1. Retrieval: the default retriever (RETRIEVAL_MODE, RERANK) on every gate set. Fails if any
   metric (Hit@1/5/10, MRR) drops, or any question leaves the top 5. Improvements pass, with a
   note to record them.
2. Guardrail replay: stored answers (data/eval/answers/) go through the current guardrails.
   Fails if any answer's flags differ from the baseline, e.g. a new false flag, or a known
   failure (ResNet misattribution, loyalty "250") no longer flagged.

An intended change in the numbers is recorded with `rag eval gate --update "reason"`, which
rewrites eval/baseline.json with the reason and the commit: every change in quality is a
deliberate, documented step. Data (database, indexes, stored answers) is local and not in git,
so on a machine without it the affected check is skipped with a message, not failed.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from rag.config import PROJECT_ROOT

BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline.json"
GATE_SETS = ("papers", "hr-demo", "injection", "gov-docs")  # gov-docs: general documents (step 21b)
METRICS = ("hit@1", "hit@5", "hit@10", "mrr")
TOP = 5
# Stored answer runs replayed through the guardrails: step 16 (labelled) and step 17.
REPLAY_RUNS = (
    "papers-20261123T062859Z-fused",
    "hr-demo-20261123T063124Z-fused",
    "papers-20261123T082735Z-v3-guard",
    "hr-demo-20261123T083001Z-v3-guard",
    "injection-20261123T081811Z-inj-v3-sanitize-false",
    "gov-docs-20261124T080146Z-step21b",  # general documents (step 21b)
)


# --- measuring ----------------------------------------------------------------------------


def retrieval_snapshot(conn, searcher) -> dict:
    """Per gate set: headline metrics and every answerable question's first gold rank."""
    from rag.eval.runner import score_questions, summarise
    from rag.eval.sets import load_set, resolve_gold, set_path

    snapshot = {}
    for name in GATE_SETS:
        questions = load_set(set_path(name))
        gold, checks = resolve_gold(conn, questions)
        broken = [c for c in checks if c.status in ("split", "missing")]
        if broken:
            raise ValueError(f"{name}: {len(broken)} gold quotes have no chunk (run `rag eval check --set {name}`)")
        rows = score_questions(searcher, questions, gold, True, {})
        overall = summarise(rows)["overall"]
        snapshot[name] = {
            "n": overall["n"],
            "metrics": {m: overall[m] for m in METRICS},
            "ranks": {r["id"]: r.get("first_rank") for r in rows if r["answerable"] and r["gold"]},
        }
    return snapshot


def replay_flags(conn, answers_dir: Path) -> dict | None:
    """Flags the current guardrails raise on each stored answer: {"run|question|attempt": [kinds]}.
    None if the stored runs are not on this machine."""
    from rag.answer.guardrails import check_answer_text
    from rag.store import db

    paths = [answers_dir / f"{run_id}.json" for run_id in REPLAY_RUNS]
    if not all(p.exists() for p in paths):
        return None
    doc_cache: dict[str, str] = {}

    def doc_text(doc_id: str) -> str:
        if doc_id not in doc_cache:
            document = db.get_document(conn, doc_id)
            title = document.title if document else ""
            doc_cache[doc_id] = f"{title} " + " ".join(s.text for s in db.get_sections(conn, doc_id))
        return doc_cache[doc_id]

    flags = {}
    for path in paths:
        run = json.loads(path.read_text())
        for record in run["questions"]:
            for attempt in record["attempts"]:
                sources = attempt["sources"]
                first = [s["chunk_ids"][0] for s in sources]
                chunk_doc = {c.chunk_id: c.doc_id for c in db.get_chunks_by_ids(conn, first)}
                source_docs = {s["n"]: chunk_doc.get(s["chunk_ids"][0], "") for s in sources}
                result = check_answer_text(
                    record["question"],
                    attempt.get("raw_answer") or attempt["answer"],
                    {s["n"]: s["text"] for s in sources},
                    source_docs,
                    {d: doc_text(d) for d in set(source_docs.values()) if d},
                )
                key = f"{run['run_id']}|{record['id']}|{attempt['run']}"
                flags[key] = sorted({f.kind for f in result.flags})
    return flags


# --- comparing ----------------------------------------------------------------------------


def compare_retrieval(baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    """(failures, improvements). Zero tolerance: any drop, any question leaving the top 5."""
    failures, improvements = [], []
    for name, base in baseline.items():
        now = current.get(name)
        if now is None:
            failures.append(f"{name}: not measured")
            continue
        for metric in METRICS:
            before, after = base["metrics"][metric], now["metrics"][metric]
            if after < before:
                failures.append(f"{name} {metric}: {before} -> {after}")
            elif after > before:
                improvements.append(f"{name} {metric}: {before} -> {after}")
        for qid, before in base["ranks"].items():
            after = now["ranks"].get(qid)
            in_before, in_after = (before or 99) <= TOP, (after or 99) <= TOP
            if in_before and not in_after:
                failures.append(f"{name} {qid}: rank {before} -> {after or '>20'} (left the top {TOP})")
            elif not in_before and in_after:
                improvements.append(f"{name} {qid}: rank {before or '>20'} -> {after} (entered the top {TOP})")
    return failures, improvements


def compare_flags(baseline: dict, current: dict) -> list[str]:
    """Any answer whose guardrail flags changed, either way."""
    failures = []
    for key in sorted(set(baseline) | set(current)):
        before, after = set(baseline.get(key, [])), set(current.get(key, []))
        if before != after:
            added, removed = sorted(after - before), sorted(before - after)
            change = ", ".join([f"+{k}" for k in added] + [f"-{k}" for k in removed])
            failures.append(f"flags changed on {key.replace('|', ' ')}: {change}")
    return failures


# --- the gate -----------------------------------------------------------------------------


def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_baseline(
    retrieval: dict,
    flags: dict | None,
    config: dict,
    reason: str,
    commit: str | None,
    previous: dict | None,
    path: Path = BASELINE_PATH,
) -> dict:
    history = (previous or {}).get("history", [])
    entry = {
        "when": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "reason": reason,
        "metrics": {name: s["metrics"] for name, s in retrieval.items()},
    }
    baseline = {
        "updated": entry["when"],
        "commit": commit,
        "reason": reason,
        "config": config,
        "retrieval": retrieval,
        "flags": flags if flags is not None else (previous or {}).get("flags"),
        "history": [*history, entry],
    }
    path.write_text(json.dumps(baseline, indent=1, ensure_ascii=False) + "\n")
    return baseline
