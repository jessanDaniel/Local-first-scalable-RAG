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
import time
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
        start = time.perf_counter()
        hits = searcher.search(question.question, MAX_K, collections)
        latency_ms = (time.perf_counter() - start) * 1000
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
            "latency_ms": round(latency_ms, 1),
            "fallback": getattr(searcher, "last_fallback", None),
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
        "latency_ms": latency_summary(rows),
        "fallbacks": sum(bool(r.get("fallback")) for r in rows),
    }


def latency_summary(rows: list[dict]) -> dict | None:
    """p50 / p95 / max search time per question (ms), all questions. Runs warm up first."""
    times = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    if not times:
        return None
    pick = lambda q: times[min(len(times) - 1, int(round(q * (len(times) - 1))))]  # noqa: E731
    return {"p50": pick(0.5), "p95": pick(0.95), "max": times[-1]}


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


def retriever_and_rerank(config: dict) -> tuple[str, str]:
    """(retriever, rerank) of a run, for runs of every age: runs before step 13 have no retriever
    (dense); runs before 15b used retriever "rerank", which meant hybrid + replace."""
    retriever = config.get("retriever", "dense")
    rerank = config.get("rerank", "replace" if retriever == "rerank" else "off")
    return ("hybrid" if retriever == "rerank" else retriever), rerank


def run_brief(run: dict) -> dict:
    """One line of a run list (the web UI's Runs table): headline metrics and what was run."""
    retriever, rerank = retriever_and_rerank(run["config"])
    overall = run["summary"]["overall"]
    return {
        "run_id": run["run_id"],
        "set": run["set"],
        "label": run.get("label"),
        "created": run["created"],
        "retriever": retriever,
        "rerank": rerank,
        "collections": sorted({q["collection"] for q in run["questions"] if q.get("collection")}),
        "n_chunks": run["config"].get("n_chunks"),
        "git_commit": run["config"].get("git_commit"),
        "n": overall.get("n", 0),
        **{m: overall.get(m) for m in ("hit@1", "hit@5", "hit@10", "mrr")},
        "latency_ms": run["summary"].get("latency_ms"),
        "fallbacks": run["summary"].get("fallbacks", 0),
    }


def print_run(run: dict) -> None:
    s, c = run["summary"], run["config"]
    print(f"run {run['run_id']}")
    retriever, rerank = retriever_and_rerank(c)
    print(
        f"  {retriever}: index {c['index_build']} ({c['index_type']}, {c['n_chunks']} chunks), "
        f"chunker {c['chunker_version']}, filter {'on' if c['collection_filter'] else 'off'}, commit {c['git_commit']}"
    )
    if retriever in ("dense", "hybrid"):
        print(f"  {c['embedding_model']}, query instruction: {c['query_instruction']!r}")
    if retriever in ("bm25", "hybrid"):
        print(f"  BM25 k1={c['bm25_k1']}, b={c['bm25_b']}, tokenizer {c['tokenizer']}")
    if retriever == "hybrid":
        print(f"  RRF k={c['rrf_k']}, {c['hybrid_candidates']} candidates from each retriever")
    if rerank != "off":
        print(
            f"  rerank {rerank}: {c['rerank_model']} on {c['rerank_device']}, "
            f"top {c['rerank_candidates']} candidates -> best {c['rerank_keep']}"
        )
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
    if s.get("latency_ms"):
        lat = s["latency_ms"]
        print(
            f"  search latency per question: p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms, max {lat['max']:.0f} ms"
        )
    if s.get("fallbacks"):
        print(f"  WARNING: reranker fell back to the base order on {s['fallbacks']} questions")
    if s["no_gold"]:
        print(f"  WARNING: no gold chunk found for {', '.join(s['no_gold'])} (run `rag eval check`)")
    misses = [q for q in run["questions"] if q["answerable"] and (q["first_rank"] or 99) > 5]
    if misses:
        print("\n  gold not in the top 5:")
        for q in sorted(misses, key=lambda q: q["first_rank"] or 99):
            rank = q["first_rank"] or f">{MAX_K}"
            print(f"    {q['id']:28} rank {rank:<4} overlap {q['overlap']}  top-1 from {q['top1_doc']}")


def run_diff(a: dict, b: dict) -> dict:
    """What changed from run A to run B, as data: the config keys that differ, every metric with its
    delta, latency, and the questions whose first gold rank moved (+ = better in B), with the top-k
    lines they crossed. `compare_runs` prints it; the web UI shows the same dict."""
    if a["set"] != b["set"]:
        raise ValueError(f"runs are of different sets ({a['set']} vs {b['set']}): nothing to compare")
    config_changes = [
        {"key": k, "a": a["config"].get(k), "b": b["config"].get(k)}
        for k in sorted(set(a["config"]) | set(b["config"]))
        if a["config"].get(k) != b["config"].get(k) and k not in ("git_commit",)
    ]
    sa, sb = a["summary"]["overall"], b["summary"]["overall"]
    metrics_rows = []
    for metric in METRIC_ROWS:
        va, vb = sa.get(metric), sb.get(metric)
        delta = None if va is None or vb is None else (vb - va if metric.startswith("hit@") else round(vb - va, 4))
        metrics_rows.append({"metric": metric, "a": va, "b": vb, "delta": delta})
    latency = []
    la, lb = a["summary"].get("latency_ms"), b["summary"].get("latency_ms")
    if la and lb:
        latency = [{"key": key, "a": la[key], "b": lb[key], "delta": lb[key] - la[key]} for key in ("p50", "p95")]
    ranks_a = {q["id"]: q.get("first_rank") for q in a["questions"] if q["answerable"]}
    flips = []
    for q in b["questions"]:
        if not q["answerable"] or q["id"] not in ranks_a:
            continue
        ra, rb = ranks_a[q["id"]], q.get("first_rank")
        if ra != rb:
            change = (ra or MAX_K + 1) - (rb or MAX_K + 1)
            crossed = [k for k in (1, 5, 10) if ((ra or 99) <= k) != ((rb or 99) <= k)]
            flips.append({"id": q["id"], "a": ra, "b": rb, "change": change, "crosses": crossed})
    flips.sort(key=lambda f: -abs(f["change"]))
    ids_b = {q["id"] for q in b["questions"]}
    return {
        "a": a["run_id"],
        "b": b["run_id"],
        "set": a["set"],
        "n": {"a": sa["n"], "b": sb["n"]},
        "config_changes": config_changes,
        "metrics": metrics_rows,
        "latency_ms": latency,
        "flips": flips,
        "only_in_a": sorted(set(ranks_a) - ids_b),
        "only_in_b": sorted({q["id"] for q in b["questions"] if q["answerable"]} - set(ranks_a)),
        "max_k": MAX_K,
    }


def compare_runs(a: dict, b: dict) -> list[str]:
    """`rag eval compare`: run_diff as text."""
    d = run_diff(a, b)
    lines = [f"A = {d['a']}", f"B = {d['b']}"]
    for change in d["config_changes"]:
        lines.append(f"config {change['key']}: {change['a']!r} -> {change['b']!r}")
    lines.append(f"\n{'metric':10} {'A':>14} {'B':>14} {'B - A':>8}")
    for row in d["metrics"]:
        metric, va, vb = row["metric"], row["a"], row["b"]
        delta = f"{row['delta']:+d}" if metric.startswith("hit@") else f"{vb - va:+.3f}"
        lines.append(f"{metric:10} {fmt(metric, va, d['n']['a']):>14} {fmt(metric, vb, d['n']['b']):>14} {delta:>8}")
    for row in d["latency_ms"]:
        key = row["key"]
        lines.append(f"{'latency ' + key:10} {row['a']:>11.0f} ms {row['b']:>11.0f} ms {row['delta']:>+6.0f} ms")
    if d["flips"]:
        lines.append(f"\nquestions whose first gold rank changed ({len(d['flips'])}; + = better in B):")
        for f in d["flips"]:
            mark = f"  crosses top-{','.join(map(str, f['crosses']))}" if f["crosses"] else ""
            ra = f["a"] or ">" + str(MAX_K)
            rb = f["b"] or ">" + str(MAX_K)
            lines.append(f"  {f['change']:+3d}  {f['id']:28} {ra:>4} -> {rb:<4}{mark}")
    else:
        lines.append("\nno question changed rank")
    return lines
