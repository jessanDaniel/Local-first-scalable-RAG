# Step 20 — request logs (observability)

2026-11-23. **No new dependency:** JSON lines and the standard library. Text logging is **on** (the
user's choice for development; to be revisited for the packaged release).

## What was built

| Piece | File | What it does |
|---|---|---|
| Record + log | `src/rag/requestlog.py` | `Trace` builds one record per request (`searched`, `prepared`, `generated`, `answered`, `failed`); `RequestLog` appends it to `data/logs/requests.jsonl`, thread-safe, rotated by size (`requests.1.jsonl` …), keeps the last 500 in memory; a failed write is counted, never fails the request |
| CLI | `src/rag/search.py`, `src/rag/ask.py` | `rag search` and `rag ask` write the record; `rag ask` prints `request <id> (rag logs show <id>)` |
| API | `src/rag/api/app.py` | `/search` and `/ask` write the same record under the `X-Request-ID`; errors (404, 503, 502, mid-stream) and clients that leave mid-stream (`cancelled`) are logged too; `/health` gains a `requests` block (count, write errors, recent p50/p95 per stage) |
| Reading | `rag logs show [ID\|last] [--json]`, `rag logs stats [--last N] [--route ask\|search] [--json]` | one request as a trace; p50/p95/max per stage, error/cancel/fallback/refusal/flagged counts, Ollama prompt reading in ms/token |
| Small additions | `llm/interface.py`, `llm/ollama.py`, `answer/answerer.py`, `config.py` | `Generation.first_token_seconds` (streaming); `finish_s` in answer timings; settings `LOG_REQUESTS`, `LOG_TEXT`, `LOG_MAX_MB` (20), `LOG_BACKUPS` (5) |

Evaluation runs and the quality gate call the searcher/answerer directly, so they don't log (and
the gate is unchanged: logging alters no result).

**What a record holds:** request id, time, source (cli/api), route, status, error; settings (mode,
rerank, k, collections, prompt version, sanitising, embedder/reranker with device, index build);
the question; one number per stage (`other_ms` = total minus the named top-level stages, so the
times always add up); every candidate with its score, rank in each list (dense, BM25, fused,
rerank, final) and which numbered source it became or "dropped (budget)"; the sources; tokens
counted before sending and reported by the model; Ollama's load / prompt-reading / writing time and
time to the first token; the answer, the model's raw text when guardrails edited it, citations,
flags, warnings, the reranker fallback. `LOG_TEXT=false` keeps everything except the question,
answer and flag text.

## Proof

**Offline tests (`tests/test_requestlog.py`, 14; 323 in total):** the record's shape for search
and streamed ask, stages add up to the total, **the CLI and the API write the same fields with the
same candidates and sources**, 404 / LLM down / failing midway are logged with retrieval still in
the record, a client leaving mid-stream is logged as `cancelled` (slow fake model, 3/3 runs),
`LOG_TEXT=false` leaves no question or answer text in the file, `LOG_REQUESTS=false` writes no
file, rotation keeps N backups and reads back in order, a failing write doesn't fail the request,
`show`/`stats`/`rag logs`, Ollama's time to first token.

**Real requests against `rag serve` (`logs_step20.py`, results in `logs_step20.json`), explained
from `rag logs show` alone:**

| Request | What the log shows |
|---|---|
| `p20-normal` (sqrt d_k, 15.4 s) | Right answer from §3.2.1 (rank 1: dense 2, BM25 2, rerank 2). Slow because **Ollama loaded the model (3.3 s)** — the first request after starting Ollama — then **read 2,622 prompt tokens in 7.0 s (2.66 ms/token)**. First token at 12.2 s for the client: retrieve 1.8 (of which rerank 1.7) + load 3.3 + prompt 7.0 |
| `p20-vae-replace` (refusal) | **Retrieval failure caused by the reranker:** the §2.4 reparameterization-trick chunk isn't among the candidates at all (replace keeps the reranker's top 20; the reranker put it 22nd). The model, given §2.1 / GAN / BatchNorm chunks, correctly refused. Same failure as step 15 |
| `p20-vae-fused` (correct) | Same question, fused: §2.4 is **final rank 1** (dense 4, BM25 3 outvote rerank 22); answer cites [1] |
| `p20-refusal` (Adam GPU) | Correct refusal; Adam chunks retrieved, none says it; 0 flags |
| `p20-hr` (vacation days, 4.7 s) | Right chunk at rank 1 in every list; shorter prompt (1,827 tokens, 3.9 s to read) |
| `p20-search-vae` | Same 5 hits, scores and per-list ranks as `rag search --rerank replace --explain` on the CLI |

`other_ms` (time outside the named stages) was 0.4–2.2 ms on every record.

**Cost:** writing a record takes **0.16 ms**; warm `/search` p50 is **9.2 ms** (step 19: 9 ms).
A record is ~2.6 KB for a search and ~5.7 KB for an answer, so the 20 MB file holds ~3,500 answers
before it rotates.

## What the logs showed right away

- **Prompt reading is the bottleneck, and still slow:** p50 2.36 ms/token over these 5 answers
  (step 16: 1.0). Swap was 7.7 GB in use. Every answer now records it, so the next time it changes
  (after a reboot, with fewer apps open) the logs will show it without a separate experiment.
- **Writing:** ~23 ms/token (≈43 tokens/s), 2.4–3.2 s per answer.
- **The reranker is ~1.7–1.9 s of every answer's 1.8–2.1 s retrieval.**
- **A guardrail side effect, visible because the log keeps the raw text:** stripping citations
  from a refusal can leave a sentence without its subject ("[2][5] discuss using…" → "discuss
  using…"). Cosmetic, not fixed here.

## Limits

- One process, one file: no log shipping, alerting or dashboards (the UI, steps 21–22, will show the
  trace; `rag logs stats` is the summary).
- `/health`'s recent stats cover the last 500 requests since the server started.
- Stage times for the reranker cover scoring only; the collection filter and the passage fetch
  before it fall into `other_ms` (under 2 ms here).
