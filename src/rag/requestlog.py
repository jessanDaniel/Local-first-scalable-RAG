"""Per-request logs: one JSON line per search or answer, enough to explain it afterwards.

    rag logs show [REQUEST_ID | last]     one request as a readable trace
    rag logs stats [--last N] [--route ask|search]
                                          p50 / p95 per stage, error / fallback / refusal rates

Records go to data/logs/requests.jsonl (rotated by size: requests.1.jsonl is the previous file).
`rag search`, `rag ask` and the API (`/search`, `/ask`) write the same record through `Trace`;
evaluation runs and the quality gate don't log (they call the searcher and answerer directly).

What a record holds (schema 1):

    request_id, ts, source (cli|api), route (search|ask), status (ok|error|cancelled), error
    settings     mode, rerank, k, collections, prompt version, sanitising, models, index build
    query        the question (LOG_TEXT=false leaves it out, along with the answer and flag text)
    timings_ms   flat, one number per stage; "other_ms" = total minus the named stages
    candidates   every retrieved chunk: rank, score, its rank in each retriever list, and which
                 numbered source it became (or "dropped" for the token budget)
    sources      what the model read: number, document, chunks, tokens
    llm          provider, model, tokens counted before sending and reported after, Ollama's
                 load / prompt-reading / writing seconds, time to the first token
    answer       text, refusal, citations, guardrail flags, warnings, retrieval fallback

No new dependency: JSON lines and the standard library. One process on one Mac doesn't need a
metrics collector; `rag logs stats` reads the same file.
"""

import json
import statistics
import sys
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = 1

# Retriever timing keys -> the record's stage names. Dense and BM25 alone report "search_ms";
# inside hybrid they are prefixed already.
STAGE_NAMES = {
    "dense": {"embed_query_ms": "dense_embed_ms", "search_ms": "dense_search_ms"},
    "bm25": {"search_ms": "bm25_search_ms"},
    "hybrid": {"dense_embed_query_ms": "dense_embed_ms"},
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def retrieval_stages(base_mode: str, timings: dict[str, float]) -> dict[str, float]:
    """A searcher's last_timings with one name per stage whatever the mode."""
    names = STAGE_NAMES.get(base_mode, {})
    return {names.get(key, key): round(ms, 2) for key, ms in timings.items()}


def ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


def retrieval_settings(searcher) -> dict:
    """Which models and index a searcher used: what changes a result besides the query."""
    out: dict = {}
    base = searcher.base if searcher.mode == "rerank" else searcher
    if searcher.mode == "rerank":
        out["rerank_model"] = searcher.reranker.model_name
        out["rerank_device"] = searcher.reranker.device
        out["rerank_candidates"] = searcher.candidates
    dense = base if base.mode == "dense" else getattr(base, "dense", None)
    if dense is not None:
        out["embedding_model"] = dense.embedder.model_name
        out["embedding_device"] = dense.embedder.device
        out["index_build"] = dense.index.manifest.build_id
    if base.mode == "hybrid":
        out["hybrid_candidates"] = base.candidates
    return out


class Trace:
    """Builds one record while a request runs; `record()` returns it with the totals filled in."""

    def __init__(
        self,
        source: str,
        route: str,
        query: str,
        mode: str,
        rerank: str,
        collections: list[str] | None,
        request_id: str | None = None,
        k: int | None = None,
    ):
        self.started = time.perf_counter()
        self.data: dict = {
            "schema": SCHEMA,
            "request_id": request_id or new_request_id(),
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "source": source,
            "route": route,
            "status": "ok",
            "error": None,
            "settings": {"mode": mode, "rerank": rerank, "k": k, "collections": collections},
            "query": query,
            "timings_ms": {},
        }
        # The named stages whose sum is compared with the total ("other_ms" is the remainder).
        self.top_level: list[str] = []

    @property
    def request_id(self) -> str:
        return self.data["request_id"]

    def time(self, stage: str, milliseconds: float | None, top_level: bool = True) -> None:
        if milliseconds is None:
            return
        self.data["timings_ms"][stage] = round(milliseconds, 2)
        if top_level and stage not in self.top_level:
            self.top_level.append(stage)

    def searched(self, searcher, hits, search_ms: float) -> None:
        """A /search or `rag search`: the hits, their ranks per list, per-stage times."""
        self.data["settings"].update(retrieval_settings(searcher))
        self.time("search_ms", search_ms)
        base_mode = searcher.base.mode if searcher.mode == "rerank" else searcher.mode
        for stage, value in retrieval_stages(base_mode, searcher.last_timings).items():
            self.time(stage, value, top_level=False)
        ranks = getattr(searcher, "last_ranks", None) or {}
        self.data["candidates"] = [candidate(h, ranks.get(h.chunk.chunk_id)) for h in hits]
        self.data["fallback"] = getattr(searcher, "last_fallback", None)

    def prepared(self, answerer, prepared) -> None:
        """The first half of an answer: retrieval, numbered sources, the prompt."""
        searcher = answerer.searcher
        llm = answerer.llm
        self.data["settings"].update(
            {
                **retrieval_settings(searcher),
                "k": answerer.retrieve_k,
                "prompt": answerer.prompt_version,
                "sanitize_sources": answerer.sanitize_sources,
                "guardrails": answerer.guardrails,
                "context_tokens": answerer.context_tokens,
            }
        )
        self.time("retrieve_ms", ms(prepared.timings["retrieve_s"]))
        base_mode = searcher.base.mode if searcher.mode == "rerank" else searcher.mode
        for stage, value in retrieval_stages(base_mode, prepared.search_timings).items():
            self.time(stage, value, top_level=False)
        self.time("assemble_ms", ms(prepared.timings["assemble_s"]))
        used = {chunk_id: s.n for s in prepared.sources for chunk_id in s.chunk_ids}
        dropped = {h.chunk.chunk_id for h in prepared.dropped_hits}
        self.data["candidates"] = [
            {
                **candidate(h, prepared.ranks.get(h.chunk.chunk_id)),
                "source": used.get(h.chunk.chunk_id),
                "dropped": h.chunk.chunk_id in dropped,
            }
            for h in prepared.hits
        ]
        self.data["sources"] = [
            {"n": s.n, "doc_id": s.doc_id, "section": s.section_path, "chunk_ids": s.chunk_ids, "tokens": s.n_tokens}
            for s in prepared.sources
        ]
        self.data["fallback"] = prepared.retrieval_fallback
        self.data["removed_instructions"] = len(prepared.removed_instructions)
        self.data["llm"] = {
            "provider": llm.provider,
            "model": llm.model,
            "prompt_tokens_counted": prepared.prompt_tokens,
            "max_answer_tokens": llm.max_tokens,
        }

    def generated(self, generation, streamed_ms_to_first_token: float | None = None) -> None:
        """What the provider reported: tokens and, from Ollama, where the time went."""
        g = generation
        self.time("generate_ms", ms(g.elapsed_seconds))
        self.time("llm_load_ms", ms(g.load_seconds), top_level=False)
        self.time("llm_prompt_ms", ms(g.prompt_seconds), top_level=False)
        self.time("llm_write_ms", ms(g.write_seconds), top_level=False)
        self.time("first_token_ms", ms(g.first_token_seconds), top_level=False)
        if streamed_ms_to_first_token is not None:
            self.time("request_first_token_ms", streamed_ms_to_first_token, top_level=False)
        self.data.setdefault("llm", {}).update(
            {
                "provider": g.provider,
                "model": g.model,
                "prompt_tokens": g.prompt_tokens,
                "answer_tokens": g.completion_tokens,
            }
        )

    def answered(self, answer) -> None:
        """The second half: guardrails and citations, and the answer itself."""
        self.time("finish_ms", ms(answer.timings.get("finish_s")))
        self.data["answer"] = {
            "text": answer.text,
            "raw_text": answer.raw_text if answer.raw_text != answer.text else None,
            "refusal": answer.refusal,
            "cited": answer.citations.cited,
            "invalid_citations": answer.citations.invalid,
            "sentences": answer.citations.n_sentences,
            "cited_sentences": answer.citations.n_cited_sentences,
            "flags": [{"kind": f.kind, "detail": f.detail, "sentence": f.sentence} for f in answer.flags],
            "warnings": answer.warnings,
        }

    def failed(self, detail: str, status: str = "error", http_status: int | None = None) -> None:
        self.data["status"] = status
        self.data["error"] = detail
        if http_status is not None:
            self.data["http_status"] = http_status

    def record(self) -> dict:
        total = (time.perf_counter() - self.started) * 1000
        timings = self.data["timings_ms"]
        timings["total_ms"] = round(total, 2)
        named = sum(timings[stage] for stage in self.top_level)
        timings["other_ms"] = round(total - named, 2)
        return self.data


def candidate(hit, ranks: dict | None) -> dict:
    c = hit.chunk
    return {
        "rank": hit.rank,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "section": hit.section_path,
        "pages": [c.page_start, c.page_end],
        "score": round(hit.score, 6),
        "ranks": ranks or {},
    }


def redact(record: dict) -> dict:
    """LOG_TEXT=false: keep ids, numbers and kinds; drop what the user or the model wrote."""
    out = json.loads(json.dumps(record))
    out["query"] = None
    answer = out.get("answer")
    if answer:
        answer["text"] = answer["raw_text"] = None
        answer["flags"] = [{"kind": f["kind"], "detail": None, "sentence": None} for f in answer["flags"]]
    out["text_logged"] = False
    return out


class RequestLog:
    """Appends records to a JSON-lines file and keeps the last few in memory (for /health).
    Thread-safe: the API writes from more than one thread. A failing write never fails the
    request; it is counted and reported instead."""

    def __init__(self, path: Path | None, log_text: bool = True, max_bytes: int = 20_000_000, backups: int = 5):
        self.path = path
        self.log_text = log_text
        self.max_bytes = max_bytes
        self.backups = backups
        self.recent: deque[dict] = deque(maxlen=500)
        self.written = 0
        self.write_errors = 0
        self.last_error: str | None = None
        self.lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings) -> "RequestLog":
        path = settings.data_dir / "logs" / "requests.jsonl" if settings.log_requests else None
        return cls(path, settings.log_text, int(settings.log_max_mb * 1_000_000), settings.log_backups)

    def write(self, record: dict) -> None:
        if not self.log_text:
            record = redact(record)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.lock:
            self.recent.append(record)
            if self.path is None:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size + len(line) > self.max_bytes:
                    self.rotate()
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
                self.written += 1
            except OSError as exc:
                self.write_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

    def rotate(self) -> None:
        """requests.jsonl -> requests.1.jsonl -> ... -> requests.<backups>.jsonl (the oldest is deleted)."""
        rotated(self.path, self.backups).unlink(missing_ok=True)
        for i in range(self.backups - 1, 0, -1):
            if rotated(self.path, i).exists():
                rotated(self.path, i).rename(rotated(self.path, i + 1))
        self.path.rename(rotated(self.path, 1))


def rotated(path: Path, i: int) -> Path:
    return path.with_name(f"{path.stem}.{i}{path.suffix}")


def read_records(path: Path, backups: int = 5) -> list[dict]:
    """Oldest first, across the rotated files."""
    files = [rotated(path, i) for i in range(backups, 0, -1)] + [path]
    records = []
    for file in files:
        if file.exists():
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a line cut short by a crash
    return records


# --- summaries ----------------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def stats(records: list[dict]) -> dict:
    """Per route: counts, error / cancel / fallback / refusal / flagged rates, and p50 / p95 / max
    of every stage over the successful requests."""
    out: dict = {}
    for route in sorted({r["route"] for r in records}):
        rows = [r for r in records if r["route"] == route]
        ok = [r for r in rows if r["status"] == "ok"]
        stages: dict[str, list[float]] = {}
        for r in ok:
            for stage, value in r["timings_ms"].items():
                stages.setdefault(stage, []).append(value)
        summary = {
            "requests": len(rows),
            "ok": len(ok),
            "errors": sum(r["status"] == "error" for r in rows),
            "cancelled": sum(r["status"] == "cancelled" for r in rows),
            "fallbacks": sum(bool(r.get("fallback")) for r in rows),
            "stages_ms": {
                stage: {
                    "n": len(v),
                    "p50": round(statistics.median(v), 1),
                    "p95": round(percentile(v, 95), 1),
                    "max": round(max(v), 1),
                }
                for stage, v in stages.items()
            },
        }
        if route == "ask":
            answers = [r["answer"] for r in ok if r.get("answer")]
            summary["refusals"] = sum(a["refusal"] for a in answers)
            summary["flagged"] = sum(bool(a["flags"]) for a in answers)
            per_token = [
                r["timings_ms"]["llm_prompt_ms"] / r["llm"]["prompt_tokens"]
                for r in ok
                if r["timings_ms"].get("llm_prompt_ms") and (r.get("llm") or {}).get("prompt_tokens")
            ]
            if per_token:
                summary["prompt_reading_ms_per_token_p50"] = round(statistics.median(per_token), 2)
        out[route] = summary
    return out


# --- rag logs ------------------------------------------------------------------------------------

STAGE_ORDER = [
    "llm_check_ms",
    "search_ms",
    "retrieve_ms",
    "dense_embed_ms",
    "dense_search_ms",
    "bm25_search_ms",
    "fuse_ms",
    "rerank_ms",
    "fetch_ms",
    "assemble_ms",
    "generate_ms",
    "llm_load_ms",
    "llm_prompt_ms",
    "first_token_ms",
    "llm_write_ms",
    "request_first_token_ms",
    "finish_ms",
    "other_ms",
    "total_ms",
]
SUB_STAGES = {
    "search_ms": ["dense_embed_ms", "dense_search_ms", "bm25_search_ms", "fuse_ms", "rerank_ms", "fetch_ms"],
    "retrieve_ms": ["dense_embed_ms", "dense_search_ms", "bm25_search_ms", "fuse_ms", "rerank_ms", "fetch_ms"],
    "generate_ms": ["llm_load_ms", "llm_prompt_ms", "first_token_ms", "llm_write_ms"],
}


def per_token(stage: str, t: dict, llm: dict | None) -> str:
    """Ollama's prompt reading and writing per token: the numbers that showed the step-19 slowdown."""
    key = {"llm_prompt_ms": "prompt_tokens", "llm_write_ms": "answer_tokens"}.get(stage)
    if not key or not llm or not llm.get(key):
        return ""
    return f"   {llm[key]} tokens, {t[stage] / llm[key]:.2f} ms/token"


def show(record: dict) -> str:
    """One record as a readable trace: settings, where the time went, what was retrieved and
    used, what the model was told and wrote, and what the checks found."""
    r = record
    s = r["settings"]
    t = r["timings_ms"]
    lines = [
        f"request {r['request_id']}  {r['ts']}  {r['source']} /{r['route']}  {r['status'].upper()}"
        f"  {t.get('total_ms', 0) / 1000:.2f} s",
    ]
    if r.get("error"):
        lines.append(f"error:      {r['error']}")
    retrieval = s["mode"] + (f" + rerank {s['rerank']}" if s.get("rerank") not in (None, "off") else "")
    detail = [f"k {s.get('k')}", f"collections {', '.join(s['collections']) if s.get('collections') else 'all'}"]
    if s.get("index_build"):
        detail.append(f"index {s['index_build']}")
    lines.append(f"retrieval:  {retrieval}; " + ", ".join(detail))
    models = [
        f"{label} {s[key]} ({s.get(device, '?')})"
        for label, key, device in (
            ("embedder", "embedding_model", "embedding_device"),
            ("reranker", "rerank_model", "rerank_device"),
        )
        if s.get(key)
    ]
    if models:
        lines.append("models:     " + "; ".join(models))
    llm = r.get("llm")
    if llm:
        lines.append(
            f"answering:  {llm['provider']} {llm['model']}, prompt {s.get('prompt')}, "
            f"sanitising {'on' if s.get('sanitize_sources') else 'off'}"
            + (f" ({r['removed_instructions']} sentences removed)" if r.get("removed_instructions") else "")
        )
    lines.append(f"query:      {r['query'] if r.get('query') is not None else '(text not logged)'}")
    if r.get("fallback"):
        lines.append(f"FALLBACK:   {r['fallback']}")

    lines.append("\ntimings (ms)")
    shown = set()
    for stage in STAGE_ORDER:
        if stage not in t or stage in shown:
            continue
        shown.add(stage)
        lines.append(f"  {stage[:-3]:<22}{t[stage]:>10.1f}{per_token(stage, t, llm)}")
        for sub in SUB_STAGES.get(stage, []):
            if sub in t and sub not in shown:
                shown.add(sub)
                lines.append(f"    {sub[:-3]:<20}{t[sub]:>10.1f}{per_token(sub, t, llm)}")

    candidates = r.get("candidates") or []
    if candidates:
        lines.append(f"\ncandidates ({len(candidates)})")
        for c in candidates:
            where = ""
            if "source" in c:
                unused = "-> dropped (budget)" if c["dropped"] else "-> unused"
                where = f"-> [{c['source']}]" if c["source"] else unused
            ranks = ", ".join(f"{k} {v if v is not None else '-'}" for k, v in c["ranks"].items())
            lines.append(
                f"  {c['rank']:>2}. {c['score']:.4f} {where:20} {c['doc_id']}  §{(c['section'] or '')[:50]}"
                + (f"  ({ranks})" if ranks else "")
            )
    if llm:
        lines.append(
            f"\nprompt:     {llm.get('prompt_tokens_counted')} tokens counted before sending, "
            f"{llm.get('prompt_tokens')} reported by the model; "
            f"{len(r.get('sources') or [])} sources; answer {llm.get('answer_tokens')} tokens "
            f"(limit {llm.get('max_answer_tokens')})"
        )
    answer = r.get("answer")
    if answer:
        text = answer["text"] if answer["text"] is not None else "(text not logged)"
        lines.append(f"\nanswer{' (REFUSAL)' if answer['refusal'] else ''}:\n{text.strip()}")
        if answer.get("raw_text"):
            lines.append(f"\n(guardrails edited it; the model wrote:)\n{answer['raw_text'].strip()}")
        lines.append(
            f"\ncitations:  cited {answer['cited']}, invalid {answer['invalid_citations']}, "
            f"{answer['cited_sentences']}/{answer['sentences']} sentences cited"
        )
        for flag in answer["flags"]:
            lines.append(f"FLAG {flag['kind']}" + (f": {flag['detail']}" if flag.get("detail") else ""))
        for warning in answer["warnings"]:
            lines.append(f"WARNING {warning}")
    return "\n".join(lines)


def print_stats(summary: dict, n_records: int) -> None:
    print(f"{n_records} requests")
    for route, s in summary.items():
        print(
            f"\n/{route}: {s['requests']} requests, {s['ok']} ok, {s['errors']} errors, {s['cancelled']} cancelled, "
            f"{s['fallbacks']} reranker fallbacks"
            + (f", {s['refusals']} refusals, {s['flagged']} flagged" if route == "ask" else "")
        )
        if s.get("prompt_reading_ms_per_token_p50") is not None:
            print(f"  prompt reading: p50 {s['prompt_reading_ms_per_token_p50']} ms/token")
        print(f"  {'stage':<24}{'n':>5}{'p50':>10}{'p95':>10}{'max':>10}")
        order = {stage: i for i, stage in enumerate(STAGE_ORDER)}
        for stage, v in sorted(s["stages_ms"].items(), key=lambda kv: order.get(kv[0], 99)):
            print(f"  {stage[:-3]:<24}{v['n']:>5}{v['p50']:>10.1f}{v['p95']:>10.1f}{v['max']:>10.1f}")


def run(parsed_args) -> int:
    from rag.config import get_settings

    settings = get_settings()
    path = settings.data_dir / "logs" / "requests.jsonl"
    records = read_records(path, settings.log_backups)
    if not records:
        print(f"no requests logged yet in {path}")
        return 1
    if parsed_args.action == "show":
        wanted = parsed_args.request_id or "last"
        if wanted == "last":
            matches = records[-1:]
        else:
            matches = [r for r in records if r["request_id"].startswith(wanted)]
        if not matches:
            print(f"no request {wanted} in {path}")
            return 1
        if parsed_args.json:
            print(json.dumps(matches[-1], indent=1, ensure_ascii=False))
        else:
            print(show(matches[-1]))
        return 0
    if parsed_args.route:
        records = [r for r in records if r["route"] == parsed_args.route]
    records = records[-parsed_args.last :]
    summary = stats(records)
    if parsed_args.json:
        print(json.dumps(summary, indent=1))
    else:
        print_stats(summary, len(records))
    return 0


if __name__ == "__main__":
    from rag.main import run_module

    sys.exit(run_module(["logs"]))
