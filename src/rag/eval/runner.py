"""Run a question set through retrieval, score it, save the run, compare two runs.

A run is one JSON file in data/eval/runs/ holding everything needed to trust or
redo it: the configuration (index build and type, embedding model, query
instruction, chunker, collection filter, k, git commit, set file hash), the
summary metrics, and per question the gold chunks, the ranked top-k and the
rank of the first gold chunk. Retrieval has no LLM in it, so the same
configuration gives the same numbers on every run.
"""

import json
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from rag.config import PROJECT_ROOT
from rag.eval import metrics
from rag.eval.sets import EvalQuestion, word_overlap

KS = (1, 5, 10, 20)
MAX_K = max(KS)


def git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def score_questions(
    searcher, questions: list[EvalQuestion], gold: dict[str, list[str]], use_filter: bool, gold_texts: dict[str, str]
) -> list[dict]:
    """gold_texts: question id -> text of its first gold chunk (for the word-overlap score)."""
    rows = []
    for question in questions:
        collections = [question.collection] if use_filter else None
        hits = searcher.search(question.question, MAX_K, collections)
        ranked = [hit.chunk.chunk_id for hit in hits]
        gold_ids = gold.get(question.id, [])
        gold_set = set(gold_ids)
        row = {
            "id": question.id,
            "type": question.type,
            "tags": question.tags,
            "collection": question.collection,
            "answerable": question.answerable,
            "gold": gold_ids,
            "ranked": ranked,
            "scores": [round(hit.score, 6) for hit in hits],
            "top1_doc": hits[0].chunk.doc_id if hits else None,
            "top1_score": round(hits[0].score, 6) if hits else None,
        }
        if question.answerable:
            row["first_rank"] = metrics.first_rank(ranked, gold_set)
            text = gold_texts.get(question.id)
            row["overlap"] = round(word_overlap(question.question, text), 2) if text else None
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    """Metrics over answerable questions, overall and per type; top-1 scores for both kinds."""

    def block(selected: list[dict]) -> dict:
        if not selected:
            return {"n": 0}
        out: dict = {"n": len(selected)}
        for k in KS:
            out[f"hit@{k}"] = sum(metrics.hit_at(r["ranked"], set(r["gold"]), k) for r in selected)
        for k in (10, 20):
            out[f"recall@{k}"] = round(
                statistics.mean(metrics.recall_at(r["ranked"], set(r["gold"]), k) for r in selected), 4
            )
        out["mrr"] = round(statistics.mean(metrics.reciprocal_rank(r["ranked"], set(r["gold"])) for r in selected), 4)
        out["ndcg@10"] = round(statistics.mean(metrics.ndcg_at(r["ranked"], set(r["gold"]), 10) for r in selected), 4)
        return out

    answerable = [r for r in rows if r["answerable"] and r["gold"]]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in answerable:
        by_type[row["type"]].append(row)
    unanswerable = [r for r in rows if not r["answerable"]]

    def mean_top1(selected: list[dict]) -> float | None:
        scores = [r["top1_score"] for r in selected if r["top1_score"] is not None]
        return round(statistics.mean(scores), 4) if scores else None

    return {
        "overall": block(answerable),
        "by_type": {name: block(selected) for name, selected in sorted(by_type.items())},
        "no_gold": [r["id"] for r in rows if r["answerable"] and not r["gold"]],
        "unanswerable": {"n": len(unanswerable), "mean_top1_score": mean_top1(unanswerable)},
        "answerable_mean_top1_score": mean_top1(answerable),
    }


def new_run(set_name: str, config: dict, rows: list[dict], label: str | None = None) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{set_name}-{stamp}" + (f"-{label}" if label else "")
    return {
        "run_id": run_id,
        "set": set_name,
        "label": label,
        "created": stamp,
        "config": config,
        "summary": summarise(rows),
        "questions": rows,
    }


def save_run(run: dict, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False))
    return path


def list_runs(runs_dir: Path, set_name: str | None = None) -> list[Path]:
    runs = []
    for path in runs_dir.glob("*.json"):
        run = json.loads(path.read_text())
        if set_name is None or run["set"] == set_name:
            runs.append((run["created"], path.name, path))
    return [path for _, _, path in sorted(runs)]


def load_run(ref: str, runs_dir: Path) -> dict:
    """A run by file path, run id, or SET:latest / SET:latest~1 (the one before) ..."""
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return json.loads(path.read_text())
    if ":latest" in ref:
        set_name, latest = ref.split(":", 1)
        back = int(latest.split("~")[1]) if "~" in latest else 0
        runs = list_runs(runs_dir, set_name)
        if back >= len(runs):
            raise FileNotFoundError(f"only {len(runs)} runs of set {set_name!r} in {runs_dir}")
        return json.loads(runs[-1 - back].read_text())
    if ref.startswith("latest"):
        raise ValueError(f"{ref!r} is ambiguous across sets: use SET:{ref}, e.g. papers:{ref}")
    path = runs_dir / f"{ref}.json"
    if not path.exists():
        raise FileNotFoundError(f"no run {ref} in {runs_dir}")
    return json.loads(path.read_text())


# --- reports ------------------------------------------------------------------

METRIC_ROWS = ["hit@1", "hit@5", "hit@10", "hit@20", "recall@10", "recall@20", "mrr", "ndcg@10"]


def fmt(metric: str, value, n: int) -> str:
    if value is None:
        return "-"
    if metric.startswith("hit@"):
        return f"{value}/{n} ({value / n:.0%})"
    return f"{value:.3f}"


def print_run(run: dict) -> None:
    s, c = run["summary"], run["config"]
    print(f"run {run['run_id']}")
    print(
        f"  index {c['index_build']} ({c['index_type']}, {c['n_chunks']} chunks), {c['embedding_model']}, "
        f"chunker {c['chunker_version']}, filter {'on' if c['collection_filter'] else 'off'}, commit {c['git_commit']}"
    )
    print(f"  query instruction: {c['query_instruction']!r}")
    overall = s["overall"]
    print(f"\n  answerable questions: {overall['n']}")
    for metric in METRIC_ROWS:
        print(f"  {metric:10} {fmt(metric, overall.get(metric), overall['n'])}")
    print(f"\n  {'type':20} {'n':>3} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'mrr':>6}")
    for name, b in s["by_type"].items():
        print(f"  {name:20} {b['n']:>3} {b['hit@1']:>7} {b['hit@5']:>7} {b['hit@10']:>7} {b['mrr']:>6.3f}")
    u = s["unanswerable"]
    print(
        f"\n  mean top-1 score: answerable {s['answerable_mean_top1_score']}, unanswerable {u['mean_top1_score']} "
        f"({u['n']} questions)"
    )
    if s["no_gold"]:
        print(f"  WARNING: no gold chunk found for {', '.join(s['no_gold'])} (run `rag eval check`)")
    misses = [q for q in run["questions"] if q["answerable"] and (q["first_rank"] or 99) > 5]
    if misses:
        print("\n  gold not in the top 5:")
        for q in sorted(misses, key=lambda q: q["first_rank"] or 99):
            rank = q["first_rank"] or f">{MAX_K}"
            print(f"    {q['id']:28} rank {rank:<4} overlap {q['overlap']}  top-1 from {q['top1_doc']}")


def compare_runs(a: dict, b: dict) -> list[str]:
    lines = [f"A = {a['run_id']}", f"B = {b['run_id']}"]
    if a["set"] != b["set"]:
        raise ValueError(f"runs are of different sets ({a['set']} vs {b['set']}): nothing to compare")
    changed = {
        k: (a["config"].get(k), b["config"].get(k))
        for k in set(a["config"]) | set(b["config"])
        if a["config"].get(k) != b["config"].get(k) and k not in ("git_commit",)
    }
    for key, (va, vb) in sorted(changed.items()):
        lines.append(f"config {key}: {va!r} -> {vb!r}")
    sa, sb = a["summary"]["overall"], b["summary"]["overall"]
    lines.append(f"\n{'metric':10} {'A':>14} {'B':>14} {'B - A':>8}")
    for metric in METRIC_ROWS:
        va, vb = sa.get(metric), sb.get(metric)
        if metric.startswith("hit@"):
            delta = f"{vb - va:+d}"
        else:
            delta = f"{vb - va:+.3f}"
        lines.append(f"{metric:10} {fmt(metric, va, sa['n']):>14} {fmt(metric, vb, sb['n']):>14} {delta:>8}")

    ranks_a = {q["id"]: q.get("first_rank") for q in a["questions"] if q["answerable"]}
    flips = []
    for q in b["questions"]:
        if not q["answerable"] or q["id"] not in ranks_a:
            continue
        ra, rb = ranks_a[q["id"]], q.get("first_rank")
        if ra != rb:
            key = (ra or MAX_K + 1) - (rb or MAX_K + 1)
            flips.append((key, q["id"], ra, rb))
    if flips:
        lines.append(f"\nquestions whose first gold rank changed ({len(flips)}; + = better in B):")
        for key, qid, ra, rb in sorted(flips, key=lambda f: -abs(f[0])):
            crossed = [k for k in (1, 5, 10) if ((ra or 99) <= k) != ((rb or 99) <= k)]
            mark = f"  crosses top-{','.join(map(str, crossed))}" if crossed else ""
            lines.append(f"  {key:+3d}  {qid:28} {ra or '>' + str(MAX_K):>4} -> {rb or '>' + str(MAX_K):<4}{mark}")
    else:
        lines.append("\nno question changed rank")
    return lines
