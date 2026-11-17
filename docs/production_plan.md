# Production RAG for AI and software research papers: transition plan

Drafted 2026-11-11. Covers building the system. Evaluation is left out on purpose and comes
after the build (see §8).

---

## 0. The short version

The target is the architecture in the reference diagram, built locally and at $0:

```text
                ┌─ Dense ANN (FAISS HNSW, later quantized) ─ top 500 ─┐
query ─ parse ──┤                                                     ├─ RRF ─ 100 ─ cross-encoder ─ 20 ─ context (6–8) ─ LLM ─ cited answer
                └─ BM25 (Tantivy: title / abstract / body fields) ──── top 500 ─┘
```

What the repo has today is one stage of that: dense search over Chroma, on two short HR `.txt`
files, using whitespace chunks with no structure. The pieces worth keeping are the **interfaces
and the habits**: `retrieve()` that knows nothing about the LLM, the `LLM` interface, deterministic
prompt building, and a debug view for every stage. Almost every implementation file gets
replaced or rewritten. That is expected, not a failure of Phases 1–15.

The build runs in five stages. Each one leaves a working system:

| Stage | Result | Corpus size |
|---|---|---|
| 1. Foundation + thin vertical slice | `rag ask` answers with citations from real papers (dense only) | ~200 papers |
| 2. Hybrid retrieval + reranking | the full diagram pipeline, with a debug view per stage | ~1k papers |
| 3. Service + observability | FastAPI service, streaming, per-request traces and timings | ~1k papers |
| 4. Scale + index engineering | incremental ingest, index versions, quantization, load test | 25k–50k papers |
| 5. Deployment + access control | native service (no Docker), API keys, private collections | same |

---

## 1. What I found in the current code

### 1.1 A bug that already affected your results: silent embedding truncation

`all-MiniLM-L6-v2` has `max_seq_length = 256` word-piece tokens. Anything past that is
**dropped silently** before embedding. I measured it on `employee_handbook`:

| Chunk size (words) | Tokens per chunk | Chunks truncated |
|---|---|---|
| 150 | 138–213 | 0 / 15 |
| 300 | 138–411 | **7 / 8** |

The late-arrivals sentence (`"three late arrivals within…"`) starts at **token 349 of 411** in
`employee_handbook_0` at 300/0. That position is never embedded. This explains Phase 14's "late
arrivals falls to rank 6–7 at chunk size 300" better than any chunk-size story: the needle was
in the chunk text but invisible to the vector. It also means `.env.example`'s
`CHUNK_SIZE_TOKENS=500` would truncate every chunk.

For research papers this matters even more (dense, jargon-heavy text tokenizes to more tokens
per word). **Rule for the new system:** chunk size is measured in the embedding model's own
tokens, and the chunker fails loudly if a chunk exceeds `model.max_seq_length`.

Worth adding one line to the Phase 14 notes so the finding isn't lost.

### 1.2 File-by-file

| File | State | Problem for production | Plan |
|---|---|---|---|
| `src/ingestion/loader.py` | `.txt` only; `document_id = path.stem` | No PDF. Filename IDs collide. No metadata (title, authors, year, arXiv ID, category). | Replace with paper acquisition + PDF parsing (§4.1–4.2) |
| `src/ingestion/chunker.py` | `text.split()` into word windows | Destroys paragraphs, sections, line breaks. No character offsets, so no exact citations. Word count, not tokens (§1.1). `chunk_id = doc_pos` changes when the chunker changes. | Rewrite: structure-aware, token-sized, offsets, content-hash IDs (§4.3) |
| `src/embeddings/embedder.py` | Loads model; `encode()` with defaults | No batch size or device, no query/passage prefixes, no length check, model name defaulted in code | Keep the class shape, add those (§4.5) |
| `src/retrieval/similarity.py` | Manual cosine + top-K; also defines `ScoredChunk` | Library module edits `sys.path` at import. Shared data class lives in an experiment module. | Move `ScoredChunk` to `models.py`. Keep `top_k()` as the exact-search reference for checking ANN recall. |
| `src/retrieval/vector_store.py` | Chroma wrapper | `add()` not `upsert()` (re-index fails or skips). No batching (Chroma caps batch size). No delete. One collection per doc. No quantization or HNSW tuning exposed. No BM25. | Replace with FAISS dense index behind a `DenseIndex` interface (§4.5) |
| `src/retrieval/retriever.py` | `index_chunks()` + `retrieve(query, k)` | Indexing and querying in one class. Dense only. No filters. | Split: `Indexer` (offline) and `Retriever` (online pipeline of stages) |
| `src/generation/context_builder.py` | Template + chunk-ID labels | No citation instruction, no token budget, no dedupe. "Say you don't know" caused the Phase 14 over-refusal. | Rewrite with numbered sources, budget, citation checks (§5.3) |
| `src/llm/interface.py` | `generate(prompt) -> dict` | Untyped dict. No streaming, no token counts, no system message. | Typed `Generation` result; add `stream()` |
| `src/llm/ollama.py` | Works | URL hard-coded (ignores `OLLAMA_BASE_URL`). Model `llama3.2:1b` hard-coded as default (breaks your own §2 rule). **No request timeout**, so a stuck call hangs forever. No token counts (Ollama returns `prompt_eval_count`/`eval_count`). | Fix all four |
| `src/llm/groq.py` | Works | `load_dotenv()` at import. No timeout. Retry lives in an experiment script, not the client. Token usage ignored. | Same fixes; retry with backoff on 429 inside the client |
| `src/config/` | Empty | `LLM_PROVIDER` never read; `.env.example` (3b model, 500-token chunks) disagrees with code defaults | `pydantic-settings` config + provider factory (§3) |
| `src/main.py` | Stub | No entry point | CLI (`rag ingest / index / search / ask`) then FastAPI |
| `tests/` | Empty | Nothing guards a refactor | Unit tests from Stage 1 |
| `pyproject.toml` | `package = false` | That's why every script needs the `sys.path` hack | Make it a real package; drop all hacks |
| Repo | **Not a git repository** | No history, no rollback, nothing to show on a resume | `git init` is step one |
| `AGENTS-learning-companion.md` | 537 lines, AGENTS.md with escaped markdown | Stale duplicate; agents may read the wrong one | Delete or move to `docs/archive/` (your call) |
| `chroma_db/` | 6.2 MB of experiment collections | Not used by the new system | Delete once Stage 1 works |
| `evaluations/*` | Phase 1–16 scripts, CSVs, notes | Tied to HR docs and the old modules | Move to `experiments/learning/`, freeze. They stay as the record. |
| `evaluations/eval_questions.jsonl` | 30 HR questions | Wrong domain for papers | Kept as history; a papers eval set comes later (§8) |

---

## 2. Target design decisions

Each is a recommendation with the reason. Items marked **confirm** need your choice before the
stage that uses them.

### 2.1 Corpus — **confirm**

**arXiv, `cs.CL`, `cs.LG`, `cs.AI`, `cs.SE`, `cs.IR`, 2022 onward.** Start at ~200 papers, grow to
1k, then 25k–50k.

- Metadata: arXiv API / OAI-PMH (free, rate limit about one request every 3 seconds), or the free
  arXiv metadata snapshot on Kaggle for bulk lists.
- PDFs: `export.arxiv.org`, throttled. Skip the requester-pays S3 bulk bucket (costs money).
- Licensing: most arXiv papers grant arXiv, not you, the right to redistribute. Indexing for
  personal use is fine. A public demo should show snippets plus a link to `arxiv.org/abs/<id>`,
  not serve full PDFs.

Rough size: 50k papers × ~50 chunks ≈ **2.5M chunks**.

### 2.2 Parsing: PDF, with GROBID run natively (no Docker)

Decided 2026-11-17: **PDF is the primary source.** arXiv's HTML version only exists for arXiv
(mostly papers since late 2023), while a general research-paper RAG has to read PDFs from any
source: ACL Anthology, conference proceedings, and Stage 5 uploads. Building on PDF from the
start means one parsing path for everything.

Research PDFs are two-column, with equations, tables, figure captions, running headers, footnotes,
and a reference list. A plain text extractor interleaves columns and mixes references into body
text, so the parser has to understand scholarly layout. Docker is not available here, so it must
run as a native process.

| Option | Quality | Speed | Without Docker |
|---|---|---|---|
| **GROBID** (primary) | Built for scholarly PDFs: title, authors, abstract, section tree, paragraphs, figure/table captions, parsed references, page coordinates for every paragraph | ~1–3 s/paper per thread (estimate, measure it) | Java service run from its Gradle wrapper with the JDK 17 already installed (`/opt/homebrew/opt/openjdk@17`); no change to the work Java setup. Apple Silicon support for its native libraries is checked first. |
| **Docling** (backup if GROBID won't run on arm64) | Good layout, reading order and tables; MIT license | Slower: seconds per page, uses MPS | `uv add docling`; torch is already installed |
| **PyMuPDF** (last-resort fallback) | Weak structure; exact page text | Very fast | `uv add pymupdf`; AGPL (fine for an open-source repo) |

Plan: **GROBID first.** When GROBID fails or its output fails the quality gates, fall back to
PyMuPDF and mark `parse_status=suspect`. If GROBID cannot run natively on Apple Silicon, Docling
becomes the primary parser instead. Every parser sits behind one interface
(`parse(pdf_path) -> Paper + list[Section]`), the same pattern as the LLM interface, so
swapping or adding one is a new module, not a rewrite. The parser used is recorded on every
paper.

Citations carry the page number: `[3] Hu et al. 2021 (2106.09685) §4.2, p.5`, linking to
`arxiv.org/pdf/<id>#page=5` for arXiv papers.

### 2.3 Storage: SQLite is the source of truth; indexes are rebuildable

```text
papers(arxiv_id PK, version, title, authors_json, abstract, categories, published, pdf_sha256,
       parser, parse_status, ingested_at)
sections(id PK, arxiv_id FK, ord, heading, level, page_start)
chunks(chunk_id PK = sha1(arxiv_id|section|text), arxiv_id FK, section_id FK, ord,
       chunk_type [body|abstract|caption|table], text, char_start, char_end,
       page_start, page_end, n_tokens, chunker_version)
index_builds(build_id PK, kind [dense|sparse], embedding_model, chunker_version, params_json,
             n_chunks, created_at, status)
```

The FAISS and Tantivy indexes only hold chunk IDs (FAISS: int64 row → `chunk_id` mapping table).
Text, metadata and citations always come from SQLite. Any index can be deleted and rebuilt from
it. This replaces Chroma's "store everything in the vector DB" model.

### 2.4 Dense index: FAISS (replaces Chroma)

Chroma did its job in the learning track, but it hides the knobs the diagram is about (HNSW
parameters, quantization) and has no BM25. FAISS is a library, not a framework: every index type
is explicit, which fits the project's "manual before magic" rule.

Path, in order, each step checked against the step before it:

1. `IndexFlatIP`: exact search. Same maths as your Phase 6 `top_k()`. Serves as ground truth.
2. `IndexHNSWFlat` (M=32, efConstruction=200, efSearch tunable): the diagram's ANN box.
   Measure **ANN recall@k against Flat** on sample queries. That is an index correctness check,
   not the deferred answer-quality eval.
3. Quantization, only when memory forces it (Stage 4):
   - `SQ8` (scalar, float32 → int8, 4× smaller). **This is what the diagram's "float32 → int8"
     label shows.**
   - `PQ` (product quantization: split the vector into sub-spaces S1…Sm, store one codebook ID per
     sub-space, 32×+ smaller). **This is what the diagram's "S1 S2 S3" boxes show.** The diagram
     puts both under one "Product Quant." label; they are different techniques.
   - With PQ, re-score the top candidates with full vectors (`IndexRefineFlat`) to recover
     accuracy.

Memory at 2.5M chunks (vectors only, HNSW graph adds roughly 0.3–0.6 GB):

| Dims | float32 | SQ8 | PQ (48 B/vector) |
|---|---|---|---|
| 384 | 3.8 GB | 1.0 GB | 0.12 GB |
| 768 | 7.7 GB | 1.9 GB | 0.12 GB |

On a 24 GB Mac, 384-dim float32 fits at 50k papers. Quantization becomes **necessary** around
all of arXiv CS (tens of millions of chunks). So it's built in Stage 4 as a measured experiment
(memory, latency, recall loss), not as a day-one default.

Alternative considered: **Qdrant** (normally run in Docker, which is not available here; a native binary exists; HNSW, scalar/product quantization, sparse
vectors and built-in RRF hybrid). It is the better pick if you'd rather run a database service
than own index files. The catch is that its built-in hybrid query hides fusion inside the
server. If you choose Qdrant, use its dense and sparse searches separately and do RRF yourself.

### 2.5 Embedding model — **confirm**

MiniLM's 256-token limit and general-web training data are a poor fit for papers. Candidates,
all free and local:

| Model | Dims | Max tokens | Notes |
|---|---|---|---|
| **`BAAI/bge-small-en-v1.5`** (recommended start) | 384 | 512 | Strong for its size. Uses a query instruction prefix. Same dims as today. |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | Better, 2× memory, ~3× slower to embed |
| `nomic-ai/nomic-embed-text-v1.5` | 768 (Matryoshka, can cut to 256/512) | 8192 | Long context; needs `search_query:` / `search_document:` prefixes |
| `allenai/specter2` | 768 | 512 | Trained on citations; good for paper-to-paper similarity, weaker for question → passage |

Pick one, record it in `index_builds`, and treat a model change as a full re-index. Prefixes
(query vs passage) must live inside `Embedder`, never in callers.

### 2.6 Sparse index: Tantivy (BM25)

Research questions are full of exact tokens that embeddings blur: `LoRA`, `QLoRA`, `RLHF`,
`DPO`, `GPT-4`, `ResNet-50`, `BLEU`, `MMLU`, `Table 3`, arXiv IDs, author names. That is the
reason for hybrid search here, the same lesson as Phase 5's "Java" pair.

- `tantivy` (Python bindings to a Rust Lucene-style engine): on-disk, incremental, BM25, phrase
  queries, per-field boosts (`title^3`, `abstract^2`, `section_heading^1.5`, `body^1`).
- Simpler fallback: `bm25s` (in-memory, NumPy/SciPy). Fine up to a few million chunks but no
  incremental updates.
- Tokenizer work matters more than the library: keep `GPT-4`, `fine-tuning` and `BERT-base`
  searchable both as whole terms and as parts. Write unit tests for these.

### 2.7 Fusion: Reciprocal Rank Fusion, written by hand

`score(d) = Σ 1 / (k + rank_i(d))`, k = 60. About 15 lines. Scores from BM25 and cosine are on
different scales, and RRF uses only ranks, so nothing needs calibrating. Defaults: dense top 500 +
BM25 top 500 → fused top 100. Run the two searches concurrently (thread pool; both release the
GIL in native code).

### 2.8 Reranker: cross-encoder, 100 → 20

- `BAAI/bge-reranker-base` (stronger) or `cross-encoder/ms-marco-MiniLM-L-6-v2` (faster).
- Cost: 100 query–passage pairs per query. On the M4 Pro with MPS, expect roughly 100–400 ms
  (estimate, measure it). On CPU, a lot more. The diagram's "under 100 ms p99" is a
  managed-service number for retrieval only. Locally, set a separate budget per stage
  (§6, Stage 3) and make the candidate count (100) configurable so it can be traded for latency.
- Phase 13 showed more context can hurt a small model. So: rerank to 20, then **send only the top
  6–8 (token-budgeted)** to the LLM. The 20 are still returned to the caller as "sources".

### 2.9 Generation with citations

- Context: numbered sources, each with title, arXiv ID, section, page:
  `[3] Hu et al. 2021, "LoRA…" (2106.09685) §4.2, p.5`.
- Prompt asks for claims cited inline as `[n]`. If sources are insufficient, the model says
  what is missing instead of a bare "I don't know" (this targets the Phase 14 over-refusal).
- After generation: parse the `[n]` markers, drop or flag any citation that points to a source
  that wasn't provided, and return a `citations[]` list with deep links.
- Context assembly: dedupe near-duplicate chunks, merge chunks next to each other in the same
  section, optionally expand a hit to its parent paragraph, stop at the token budget.
- Default LLM stays local (Ollama). `llama3.2:1b` failed basic arithmetic in Phase 12, so use a
  larger local default (`qwen2.5:7b-instruct` or `llama3.1:8b`; both fit in 24 GB). Groq stays
  the optional hosted provider.

### 2.10 Not using

LangChain, LlamaIndex, Pinecone and Weaviate stay out (AGENTS.md §4). Every stage above is your
own code over plain libraries (FAISS, Tantivy, sentence-transformers, FastAPI). That's also the
better resume story: you can explain every box in the diagram because you built it.

---

## 3. Target repository layout

```text
src/rag/
  config.py              pydantic-settings; one Settings object; validates provider/model at start
  models.py              Paper, Section, Chunk, Candidate, RetrievalTrace, Generation, Citation
  store/db.py            SQLite schema, migrations, repository functions
  acquire/arxiv.py       metadata listing + throttled PDF download + manifest
  parse/base.py          Parser interface: parse(pdf) -> Paper + list[Section]
  parse/grobid.py        PDF → GROBID TEI XML → sections/paragraphs/captions/refs/pages
  parse/pymupdf.py       last-resort fallback
  chunk/chunker.py       structure-aware, tokenizer-sized chunker (versioned)
  embed/embedder.py      batched, device-aware, prefix-aware
  index/dense.py         DenseIndex interface + FAISS implementations
  index/sparse.py        SparseIndex interface + Tantivy implementation
  index/builder.py       build/rebuild/increment; writes index_builds
  retrieve/fusion.py     RRF
  retrieve/rerank.py     CrossEncoderReranker
  retrieve/pipeline.py   query → dense ‖ sparse → fuse → rerank → candidates + trace
  generate/context.py    numbering, dedupe, merge, budget
  generate/prompt.py     templates
  generate/citations.py  parse + verify [n] markers
  llm/{base,ollama,groq,factory}.py
  api/app.py             FastAPI
  cli.py                 rag ingest | parse | chunk | index | search | ask | serve
  obs/                   logging, timing, traces
tests/
experiments/learning/    frozen Phase 1–16 scripts, CSVs, notes
data/                    gitignored: raw/, parsed/, rag.db, indexes/<build_id>/
docs/
```

`retrieve/pipeline.py` returns a `RetrievalTrace` (candidates and scores after each stage, with
timings). That keeps the project rule "every stage has a debug mode" working in production:
`rag search --explain` and the API's `?debug=true` both print it.

---

## 4. Ingestion pipeline (offline)

Each step is its own CLI command, can be resumed, and records status per paper in SQLite, so a
crash at paper 18,000 restarts at 18,001.

1. **Acquire** (`rag ingest --category cs.CL --since 2022-01 --limit 200`): list IDs, store
   metadata, download PDFs throttled, record sha256. Skip IDs already present.
2. **Parse** (`rag parse`): send each PDF to the local GROBID service (N workers), TEI XML →
   `Paper` + `Section`s with page numbers; PyMuPDF if GROBID fails. Quality gates: no
   body text, too few sections, too many non-alphabetic characters (a sign of a broken font map),
   or abstract missing → mark `parse_status=suspect` and try the fallback. Dump each parse to
   `data/parsed/<id>.json` for inspection (the Phase 1 "extraction stats" idea, per paper).
3. **Chunk** (`rag chunk`), rules:
   - Unit is the paragraph from the parser. Pack paragraphs into chunks of **≤ 350 tokens** (start
     value, under the 512 cap with room for the header), split long paragraphs at sentence
     breaks, overlap one sentence.
   - Never cross a section boundary (fixes the Phase 9 "Warranty chunk contains Privacy" issue).
   - Abstract is one chunk of its own. Figure and table captions are their own `chunk_type`.
     The reference list is **not** indexed as body text.
   - Contextual header for embedding and BM25, not for display:
     `"{title} | {section path}\n{text}"`. Cheap, and helps chunks like "we set it to 0.1",
     which mean nothing without their section.
   - Store `char_start/char_end/page` for citations, and `chunker_version`.
   - Fail loudly if `n_tokens > embedder.max_seq_length` (§1.1).
4. **Embed + index** (`rag index build --kind dense|sparse`): batch embed (MPS when available),
   write the FAISS index and the id map, write the Tantivy index, record the build. Indexes go in
   `data/indexes/<build_id>/`, and a `current` pointer is switched in one atomic step.

---

## 5. Query pipeline (online)

### 5.1 Steps and default numbers

| Step | Default | Configurable | Output in the trace |
|---|---|---|---|
| Query normalize | trim, collapse whitespace, extract arXiv IDs / quoted phrases | — | parsed query |
| Filters | optional `year`, `category`, `arxiv_id`, `collection` | yes | filter used |
| Dense | top 500, efSearch 128 | yes | ids, cosine, ranks, ms |
| Sparse | top 500 | yes | ids, BM25, ranks, ms |
| RRF | k=60 → 100 | yes | fused rank, which list(s) each came from |
| Rerank | 100 → 20 | yes, and can be turned off | rerank score, rank change |
| Context | top 6–8 within a token budget (e.g. 3,000) | yes | exact prompt string |
| Generate | Ollama default | provider via config | tokens in/out, model, ms |
| Citations | verify `[n]` | — | valid / invalid markers |

Filters: apply them in SQLite to get an allowed chunk-id set, then pass it to FAISS
(`IDSelector`) and Tantivy (filter query). With a very selective filter, exact search over the
allowed subset is faster than HNSW. Handle both cases.

### 5.2 Failure handling

Every external call (arXiv, Ollama, Groq) gets a timeout and bounded retries. If the reranker
fails or times out, return fused results and mark `degraded=true`. If the LLM fails, still return
the sources. A failure in one stage must not lose the work of the stages before it.

### 5.3 Response shape

```json
{
  "answer": "LoRA freezes the pretrained weights and trains low-rank matrices A and B … [1][3]",
  "citations": [{"n": 1, "arxiv_id": "2106.09685", "title": "…", "section": "4.1", "page": 4,
                 "url": "https://arxiv.org/abs/2106.09685", "chunk_id": "…"}],
  "sources": [ /* the reranked 20, with scores */ ],
  "trace_id": "…",
  "timings_ms": {"dense": 0, "sparse": 0, "fusion": 0, "rerank": 0, "llm": 0},
  "degraded": false
}
```

---

## 6. Build stages

Your preference from AGENTS.md still holds: one slice at a time; you type the core logic (chunker,
RRF, pipeline, citations); I can do the boilerplate (deps, schema, CLI wiring) when you
ask.

### Stage 1: Foundation + thin vertical slice

Goal: a real question about real papers gets a cited answer, end to end, dense only.

1. `git init`, first commit of the current state (so the learning track is in history).
2. Package layout (§3), `package = true`, remove every `sys.path` hack; move `evaluations/` to
   `experiments/learning/`; decide on `AGENTS-learning-companion.md`.
3. `config.py` + `llm/factory.py`: `LLM_PROVIDER` actually used; model names only from config;
   timeouts; token counts; typed `Generation`.
4. `models.py` + SQLite schema.
5. Run GROBID natively (Gradle wrapper, JDK 17) and parse one PDF to prove it works on Apple
   Silicon; if it can't, switch to Docling. Then acquire ~200 arXiv PDFs (cs.CL + cs.LG, 2023+),
   parse, dump to `data/parsed/<id>.json`, read 10 by hand, record the fallback rate and
   seconds per paper.
6. Chunker v2 + tests (section boundaries, token cap, offsets, deterministic IDs).
7. Embedder v2 (new model, prefixes, batching) + FAISS `IndexFlatIP`.
8. Citation-aware context + prompt + `citations.py`.
9. CLI: `rag ingest`, `rag parse`, `rag chunk`, `rag index build`, `rag search --explain`,
   `rag ask`.
10. Tooling: `ruff`, `pytest`, a pre-commit hook.

Done when: `rag ask "How does LoRA reduce the number of trainable parameters?"` returns an answer
whose citations open the right paper and section, and `--explain` shows every stage.

### Stage 2: Hybrid retrieval + reranking (the diagram)

1. Tantivy index with field boosts + tokenizer tests for the identifier cases in §2.6.
2. `fusion.py` (RRF) + tests (ties, an item in only one list, empty list).
3. Concurrent dense ‖ sparse in `pipeline.py`.
4. Switch FAISS Flat → HNSW. Measure ANN recall@100 against Flat on ~50 sample queries. Tune
   `efSearch`.
5. Cross-encoder reranker, batched, MPS, with a timeout and fallback.
6. `--explain` shows, per candidate: dense rank, BM25 rank, RRF rank, rerank rank. This is the
   view you'll use to say why a result appeared.
7. Grow the corpus to ~1k papers across 3–5 categories.

Done when: an identifier query (`"QLoRA 4-bit NormalFloat"`) and a conceptual query (`"why does
RLHF use a KL penalty"`) both come back with sensible sources, and the trace shows which list
brought each one in.

### Stage 3: Service + observability

1. FastAPI: `POST /search`, `POST /ask` (with SSE streaming), `GET /papers/{id}`, `GET /health`
   (checks DB, indexes, LLM). Pydantic request/response models. Request IDs.
2. Load models and indexes once at startup. The reranker and embedder run in a worker thread so
   they don't block the event loop.
3. Structured JSON logs, one record per request: trace ID, query hash, filters, per-stage ms,
   candidate counts, top IDs and scores, tokens, provider/model, errors, `degraded`.
4. Metrics: Prometheus counters and histograms per stage (`/metrics`). OpenTelemetry tracing is
   optional and can come later.
5. Latency budget, written down and then measured (starting targets on the M4 Pro):
   dense + sparse + RRF p95 < 50 ms; rerank(100) p95 < 400 ms; time to first token as measured.
6. A small web UI only if you want one for demos. Not needed for the core.

### Stage 4: Scale + index engineering

1. Incremental ingest: new papers append to Tantivy and to the FAISS HNSW index. Deleted or new
   versions of papers are handled by a rebuild plus an atomic `current` switch. Index builds are
   versioned (§2.3).
2. Scale to 25k–50k papers. Record GROBID throughput and fallback rate, embedding throughput, index build time,
   RAM and disk use.
3. Quantization experiment: Flat vs HNSW vs HNSW-SQ8 vs IVF-PQ (+ refine). Table of memory,
   p95 latency, ANN recall@100 against exact. Choose with numbers.
4. Embedding cache for repeated queries; reranker batch size tuning.
5. Load test (`locust`): concurrent `/search` p50/p95/p99. `/ask` is limited by the LLM, so test
   it separately.

### Stage 5: Deployment + access control

1. No Docker on this machine, so the service runs as native processes: `uv run rag serve`
   (uvicorn) and Ollama under `brew services`, kept alive by a `launchd` plist. A `Dockerfile`
   can still be written and built by CI (GitHub Actions) if you later deploy to a host that has
   Docker; it never has to run here.
2. API-key auth. Rate limiting per key.
3. Access control as collections: `public-arxiv` plus private collections for papers or notes a
   user uploads. `collection_id` is on every chunk and is applied as a required filter in both
   indexes, so a request can never see chunks outside its collections. Test that this can't be
   bypassed.
4. Upload endpoint: PDF → same parse/chunk/index path, into a private collection.
5. README: architecture diagram, design decisions with their reasons, the measured numbers from
   Stages 2–4.

---

## 7. Risks to watch

| Risk | Mitigation |
|---|---|
| GROBID mis-parses some PDFs (scanned, unusual templates), or will not run on arm64 | quality gates + PyMuPDF fallback + `parse_status` you can query; Docling as the replacement parser |
| Math-heavy text embeds poorly | BM25 helps on symbols and names. LaTeX source parsing is a later upgrade. |
| Choosing fusion/rerank settings with no evaluation | Keep every setting in config with defaults from the literature. Expect to retune once §8 exists. Don't present any default as "best". |
| Chunking changes silently invalidate indexes | `chunker_version` + `embedding_model` recorded per build; refuse to load an index whose versions don't match the DB |
| Laptop memory at scale | measure at 25k first; quantize in Stage 4 only when needed |
| arXiv throttling / terms | throttle, identify with a User-Agent, cache everything, don't serve full PDFs publicly |

---

## 8. Deferred: evaluation (after the build)

Left out as you asked. When it comes back, the building blocks are already here: the Phase 16
Hit@K/Recall@K script logic, the `RetrievalTrace` from §3, and the citation verifier from §2.9.
What's missing is a research-paper question set: ~50–100 questions with `arxiv_id` + quote
spans, the same shape as `eval_questions.jsonl`. Then: ablations (dense vs BM25 vs hybrid vs
+rerank), groundedness/citation accuracy, and regression checks in CI.

---

## 9. Decisions (agreed 2026-11-17)

1. **Corpus slice**: arXiv `cs.CL` + `cs.LG`, submitted 2023 or later, starting at 200 papers.
2. **Dense store**: FAISS index files. No Qdrant.
3. **Embedding model**: `BAAI/bge-small-en-v1.5` (384 dims, 512-token limit).
4. **Default local LLM**: `qwen2.5:7b-instruct` via Ollama. Groq stays optional.
5. **Learning-track files**: `evaluations/` moves to `experiments/learning/`;
   `AGENTS-learning-companion.md` moves to `docs/archive/` (kept, not deleted).
6. **No Docker** on this machine; **PDF is the primary source** (GROBID run natively, Docling as
   backup, PyMuPDF fallback; arXiv HTML dropped as not general enough). The service runs as
   native processes.

Progress: Stage 1 steps 1–4 done, step 5a done (GROBID 0.8.2 native on JDK 11; see
`experiments/stage1/grobid_feasibility.md`) (step 4: `rag/models.py` + SQLite store, schema v1; step 3: config, LLM factory, timeouts/retries, token counts, typed `Generation`).
Earlier: (git repo; `src/rag/` installed package, no `sys.path`
hacks; learning scripts in `experiments/learning/`; companion doc in `docs/archive/`).
