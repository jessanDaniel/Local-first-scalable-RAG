# Step 19 — the API

2026-11-23. **New dependency (approved by the user): `fastapi`** (brings `starlette`). **`uvicorn`**
was already installed as a dependency of chromadb; it's now declared directly. Streaming design:
"stream, then correct" (the user's choice).

## What was built

| Piece | File | What it does |
|---|---|---|
| Shared pipeline | `src/rag/pipeline.py` | `Pipeline`: dense (embedder + FAISS), BM25, reranker, LLM and prompt tokenizer, each loaded at most once on first use. `searcher(mode, rerank)` and `answerer(mode, rerank)` are cached per pair. `make_searcher` and `rag ask` now go through it, so the CLI and the API share one code path |
| Answer halves | `src/rag/answer/answerer.py` | `ask()` = `prepare()` (retrieve, sources, prompt, token check) → model → `finish()` (guardrails, citations, warnings). The API streams between the two halves. `Answer` now carries `ranks`, `retrieval_fallback` and `search_timings`, copied when the call is made (the searcher is shared) |
| Streaming | `src/rag/llm/interface.py`, `ollama.py`, `http.py` | `LLM.generate_stream(prompt, system, on_token)`. Ollama sends one JSON line per piece (`post_json_lines`, which retries only before the first line). Providers without streaming get a default that sends the whole text as one piece (Groq unchanged). An `on_token` that raises `GenerationCancelled` stops the request |
| Schemas | `src/rag/api/schemas.py` | Pydantic request/response models plus converters from `Hit`/`Answer` |
| Routes | `src/rag/api/app.py` | `GET /health`, `GET /collections`, `POST /search`, `POST /ask` (SSE: `sources` → `token`… → `done` or `error`; `stream: false` returns one JSON response), `X-Request-ID` on every response, `/docs` |
| Command | `src/rag/main.py` | `rag serve [--host] [--port]`; refuses anything but loopback (no authentication until step 29). Settings `API_HOST`, `API_PORT` |

**Threads.** SQLite, the embedder, BM25, FAISS and the reranker run on **one** pipeline thread,
because a SQLite connection belongs to the thread that created it and the searchers keep per-call
state. Generation (HTTP to Ollama) runs on a second thread, so a 10-second answer doesn't block
`/search`. The event loop only moves bytes. When a client disconnects mid-answer, the model is
stopped at its next token.

**Status codes.**

| Situation | Response |
|---|---|
| Pipeline failed to load | `/health` 503 `unavailable`; other routes 503 with the reason |
| Stale or missing dense index | `/health` 200 `degraded`, with the reason and "restart after rebuilding" |
| LLM down | `/ask` 503 ("/search still works"), `/health` `degraded` |
| Unknown collection | 404 |
| Bad request | 422 |
| LLM fails mid-stream | an `error` event instead of `done` |

## Proof

**Offline tests (`tests/test_api.py`, 15):** these cover health, request IDs, collections, search
(hits, ranks, filter, 404/422), parity with the pipeline called directly, the SSE event order
(streamed tokens == `raw_text`), the same prompt streamed / not streamed / CLI, guardrails on the
JSON path, LLM down (`/ask` 503, `/search` 200), failure midway (`error` event), a pipeline that
can't load, Ollama's stream parsing and missing final line, and `serve` refusing `0.0.0.0`.

**Parity with the CLI on real data and models (`api_parity.py`, results in `api_parity.json`):**

| Check | Result |
|---|---|
| `/search` hybrid, all 84 gate-set questions: same chunk IDs, same order, same scores | **84/84** |
| `/search` + fused rerank, 20 papers questions | **20/20** |
| `/ask`, 10 questions: prompt sent to the model byte-identical | **10/10** |
| same sources | **10/10** |
| same answer text | 9/10: `policy_exchange_window` added "for items of equal or lesser value"; both correct, both cite [1], no flags. This is Ollama's known temperature-0 nondeterminism (step 8), from an identical prompt |

## Latency (`api_latency.py` against a running `rag serve`, results in `api_latency.json`)

| | CLI (new process each time) | API (warm) |
|---|---|---|
| `search`, hybrid | 11.8 s | **p50 9 ms, p95 13 ms** (45 questions) |
| `search`, fused rerank | 19.3 s | p50 1.78 s, p95 1.90 s |
| server startup (cold, both models) | — | 21 s (17.8 s in the parity run) |

`/ask` (fused, 6 papers questions not asked before; medians):

| Step | Seconds |
|---|---|
| `sources` event (retrieval + rerank) | 1.8 |
| first token | 8.7 |
| `done` | 9.7 |

**Where the 8.7 s goes:** Ollama's own report says it spends **6.6–8.0 s reading the ~2,800–3,300
token prompt (2.4 ms/token)**. Step 16 measured 1.0 ms/token. With the server stopped, Ollama alone
still read at **2.3 ms/token**, so the server isn't the cause. The Mac had **10 GB of swap in use**
during these runs. This is the same slowdown step 17 recorded and left unexplained; memory pressure
is now the main suspect. Step 20's per-request logs will make this visible request by request.

**The cache trap:** the first latency run reused the parity questions and showed a first token
0.1 s after the sources. Ollama had cached those exact prompts. Measure latency only on questions
it hasn't just seen.

**Memory:** the server's physical footprint is **4.2 GB** (embedder + reranker on MPS + BM25 +
FAISS + tokenizer), next to Ollama's 4.8 GB.

## Limits

- One user at a time in practice: requests take turns on the pipeline thread, and Ollama generates
  one answer at a time. Concurrent load is step 27.
- The server keeps the index it loaded at startup. A rebuilt index needs a restart (`/health`
  says so); rebuilding without downtime is step 24.
- No authentication or rate limits (step 29), hence loopback only.
- Streamed tokens are the model's raw text. The `done` event carries the corrected text (refusal
  citations stripped) and the flags; a client must show `done.text` once it arrives.
- Starlette's TestClient warns that it prefers `httpx2`. That would be a new package, so it's not
  added; the warning is filtered in `pyproject.toml`.
