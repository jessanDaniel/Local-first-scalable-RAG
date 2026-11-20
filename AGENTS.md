# AGENTS.md — RAG Learning Companion Investigation

This file is the persistent project brief for any AI agent (Cursor, or otherwise) working in this
repository. Read it before doing anything else. It encodes the project's goals, constraints, build
order, and — importantly — **how the agent should behave** while collaborating on this project.

If a request conflicts with this file, prefer this file unless the user explicitly overrides it.

---

## 1. What this project actually is

This is **not** "build a PDF chatbot." It is a hands-on investigation of how Retrieval-Augmented
Generation systems work, where they fail, and how to measure and improve them:

```text
Learning material → Text extraction → Chunking → Embeddings → Vector search → Retrieval
          → Context construction → LLM generation → Grounded response
          → Learning interaction → Evaluation → Learner progress
```

Every stage above must remain **visible and inspectable** — never hidden behind a framework. If a
choice would hide a stage (e.g. reaching for LangChain to do retrieval), don't make that choice.

The user is a software engineer transitioning into Applied AI Engineering. They already know the
basics (LLMs, tokens, prompts, embeddings, vector DBs at a conceptual level). **Do not re-explain
fundamentals at length.** Move quickly to implementation, experimentation, debugging, trade-offs,
failure analysis, and evaluation.

**Success is not** "the demo answers questions." **Success is** the user being able to look at a
bad answer and say precisely why it happened — retrieval failure vs. generation failure vs.
chunking/config choice — and back that claim with measurements, not vibes.

### Current direction (2026-10-10) — learning track closed

The user is done with this investigation as a phase-by-phase build. Phases 1–15 are the completed
dev: the pipeline runs end to end, and the experiments that explain its failures are written up.
**Do not resume Phase 16 or walk Phases 16–25** unless the user explicitly reopens one.

The active project is now **production-grade RAG**: turn this system into a reliable,
resume-worthy service (hybrid retrieval, reranking, citations, access control, evaluation,
observability, deployment). The existing code and eval set are the foundation, not something to
rebuild. Phases 16–25 stay in §1b and §7 as a backlog that can be revisited after that work, or
folded into it when a production slice needs the same measurement.

The standing reminder to explain the Phase 16 retrieval results in plain language before anything
else is **cancelled**.

---

## 1b. Progress tracker (UPDATE THIS as phases complete)

> Agents: read this section first after the intro. Do not re-scaffold or re-explain completed
> phases. Resume at **Next up**. When a phase finishes, update the checkboxes, findings, and
> "Next up" before ending the session.

### Status (as of 2026-10-10, learning track closed)

| Phase | Name | Status |
|---|---|---|
| — | Project scaffold (`uv`, layout, `.env.example`, README stub) | ✅ Done |
| 1 | Ingestion (`.txt` loader + stats) | ✅ Done |
| 2 | Fixed-size chunker + metadata | ✅ Done |
| 3 | Chunking experiments (size × overlap + boundary check) | ✅ Done |
| 4 | Local embeddings (`sentence-transformers`) | ✅ Done |
| 5 | Embedding similarity experiments | ✅ Done |
| 6 | Manual NumPy cosine Top-K search + retrieval debug | ✅ Done |
| 7 | Chroma (after manual search) | ✅ Done |
| 8 | Query retrieval — `retrieve(query, k)`, decoupled from LLM | ✅ Done |
| 9 | Context construction (template + debug mode) | ✅ Done |
| 10 | Local LLM (Ollama) wired end-to-end | ✅ Done |
| 11 | Provider abstraction (`LLM` interface + Ollama + Groq) | ✅ Done |
| 12 | Grounding + failure experiments (mandatory) | ✅ Done |
| 13 | Top-K experiments (K ∈ {1,3,5,10}) | ✅ Done |
| 14 | Chunking × Top-K matrix | ✅ Done |
| 15 | Evaluation dataset (30 questions) | ✅ Done |
| 16 | Retrieval evaluation (Hit@K / Recall@K) | ⏸ Paused — run exists, notes not written |
| 17–25 | Generation eval → metadata → hybrid/rerank → observability → tests → README → knowledge test | ⏸ Paused — revisit only if the user asks |

**Paused on purpose (2026-10-10):** The user closed the learning track because the pipeline dev
is complete, and is turning this repo into a production-grade RAG. Phases 16–25 are not abandoned
work to finish first; they are a backlog. Reopen a phase only when the user asks. Phase 16 already
has `experiments/learning/phase16_retrieval_eval.py` and `experiments/learning/phase16_retrieval_eval.csv` (Hit@K,
Recall@K, complete, distractor). `experiments/learning/phase16_retrieval_notes.md` was never written, and
the plain-language walkthrough of those results is no longer a gate.

**Scope (2026-11-20):** the product is a generic, production-grade RAG pipeline for a single
Mac; research papers are only the first dataset. See `docs/production_plan.md` §0.1: eval v0
before Stage 2, collections now, a second (txt/Markdown) corpus, guardrails, regression gate,
index type chosen by measured recall (PQ only when size demands it), and a web UI in Stage 3.
Order: step 9 → step 10 → collections + second corpus → eval v0 → Stage 2 → Stage 3 (API + UI).

**Next up:** Production-grade RAG, following `docs/production_plan.md` (§9 decisions agreed
2026-11-17: arXiv cs.CL + cs.LG 2023+, FAISS, `bge-small-en-v1.5`, `qwen2.5:7b-instruct`, no
Docker; PDF is the primary source — GROBID run natively with JDK 17, Docling as backup,
PyMuPDF fallback; arXiv HTML dropped). Stage 1 steps 1–4 are done.
Step 3: `rag/config.py`, `rag/llm/factory.py`, timeouts/retries, typed `Generation`; live check
passed (qwen2.5:7b-instruct, 5.19s cold / 0.61s warm). Ollama is currently stopped
(`brew services start ollama` to use it). Step 4: `rag/models.py` (Paper, Section, Chunk;
content-hash `chunk_id`; chunk offsets index into `Section.text`) and `rag/store/db.py`
(SQLite at `data/rag.db`, schema v1: papers, sections, chunks, index_builds; per-paper
replace writes). Step 5a done: GROBID 0.8.2 runs natively on Apple Silicon
(default JDK 17, at `~/AI-project/tools/grobid` because a space in the path breaks it); QLoRA parsed in
~2.1s warm. Write-up: `experiments/stage1/grobid_feasibility.md`. Step 5b done: `rag/parse/`
— `base.py` (`Parser` interface, `ParsedPaper`, quality gates → ok/suspect, debug print),
`tei.py` (pure TEI → sections: numbered tree, unnumbered headings as children, headless divs
continue, appendix numbers from heading text, captions only when labelled Figure/Table, pages
from coords), `grobid.py` (HTTP client, consolidation off, retries 503), `python -m rag.parse
<pdf>` saves `data/parsed/<id>.tei.xml` + `.json`. `Section` now stores per-paragraph offsets +
pages (`Paragraph`); DB schema v2 with a v1→v2 migration. Step 5c done (10 papers, not 200 — user's
choice to start small): corpus `corpus/seed_ai_research.txt` = first arXiv papers of the "AI
research" track on fanout.sh/papers (Adam, word2vec, VAE, GAN, Seq2Seq, Distillation, DDPM,
BatchNorm, ResNet, Attention). `rag/acquire/arxiv.py` (API metadata, 3 s throttle, generic
User-Agent, PDF download with sha256, skips existing), `rag/parse/pymupdf.py` (fallback, one
section per page, always suspect), `rag/ingest.py` (`uv run python -m rag.ingest arxiv <id file>
[--limit N] [--reparse] [--report csv]`; GROBID down stops the run instead of falling back).
Result: 10 ok, 0 fallback, median 1.6 s/paper; by-hand review and two new TEI rules (appendix
left in body, junk headings) in `experiments/stage1/seed10_parse_review.md`. 65 tests.
Step 5d done (user wants a generic RAG long term, papers
first): `Paper` → `Document` with `doc_id = "<source>:<key>"` (`arxiv:1412.6980`,
`file:perceptron-1958`), `source`, `source_uri`, `metadata` (arXiv version/categories live
there); schema v3 (`documents` table; v2 has no migration — old db moved to
`data/archive/v2/`); `Parser.parse(path, doc_id)`, `ParsedDocument`; "no abstract" is no longer
a quality gate. Ingest has two sources: `uv run python -m rag.ingest arxiv <id file>` (PDFs in
`data/raw/arxiv/`) and `uv run python -m rag.ingest files <pdfs or folder>` (copied to
`data/raw/files/`, title/authors from the parse). Per-document files use `doc_file_stem()`
(`arxiv_1412.6980.json`). 72 tests. Pending from the user: Perceptron, Dropout (JMLR), AlexNet
(NeurIPS) PDFs go in `data/raw/manual/`, then `rag.ingest files data/raw/manual/`. Planned:
rerun ingest at ~200 papers before Stage 2. Step 6 done: `rag/chunk/` — `tokens.py` (`HFTokenCounter`, counts
`BAAI/bge-small-en-v1.5` tokens), `chunker.py` (`Chunker`, version `v2-350`: paragraph packing,
never crosses sections, sentence split with one-sentence overlap inside long paragraphs,
word/char hard cuts, abstract/caption/body types, exact section slices, pages from paragraphs),
`uv run python -m rag.chunk [--show DOC_ID]`. 400 chunks, max 350 tokens, median 224. New:
`uv run python -m rag.ingest tei` re-applies TEI rules to saved GROBID output (no GROBID); new
TEI rule drops footnotes GROBID files under the abstract. Notes:
`experiments/stage1/chunk_seed10.md`. 89 tests. Step 7 done: `rag/embed/embedder.py` (`Embedder`: bge-small-en-v1.5 on MPS when
available, query instruction added inside `embed_query`, passages embedded as
`passage_text()` = "title | section path\nchunk" with a 60-token header, `InputTooLongError`
instead of silent truncation), `rag/index/dense.py` (FAISS `IndexFlatIP`, versioned builds in
`data/indexes/dense/<build_id>/`, `CURRENT` pointer, `load_current` refuses stale indexes),
`rag/index/build.py` (embedding cache by sha1(model+input) in `data/embeddings/`),
`uv run python -m rag.index build|list`, `uv run python -m rag.search "<q>" [-k] [--explain]`.
400 chunks embedded in 2.4 s (170/s on MPS), query ≈ 75 ms. Smoke check (10 paraphrased
questions, not the eval): full design rank1 5/10, top3 9/10, top5 10/10; beats no-instruction and
no-header variants; one near-distractor miss (VAE reparameterization vs DDPM, a Stage 2 rerank
test case). Notes: `experiments/stage1/dense_seed10.md`. 101 tests. Before each step, pitch it
first (problem, build, why, proof, risks, out of scope). Step 8 done: `rag/answer/` (`sources.py` merge + number + 3,000-token budget
counted with the Qwen2.5 tokenizer; `prompt.py` system prompt v2; `citations.py` [n] parsing and
checks; `answerer.py` with `ContextWindowError` before calling the model),
`uv run python -m rag.ask "<q>" [--explain]`. Ollama now sends `num_ctx` 8192 and `num_predict`
600 (confirmed 100% GPU, context 8192). Smoke check, 13 questions read by hand: expected paper
cited 11/11, 8/10 answerable fully correct, 2/2 unanswerable refused; VAE question wrong (DDPM
distractors ranked above the right chunk), "why do transformers…" wording credits §3.2.1 content
to other sections (answer chunk ranked 8th of 8). Key lessons: a valid citation is not a correct
one (needs a support check in the eval); temperature 0 is not fully reproducible under Ollama
(run eval questions more than once). Notes: `experiments/stage1/answer_seed10.md`. 116 tests.
Ollama is stopped (`brew services start ollama` before `rag ask`). Step 9 done: one `rag`
command (`src/rag/main.py`) — `rag status [--json]`, `rag ingest arxiv|files|tei`, `rag parse`,
`rag chunk`, `rag index build|list`, `rag search`, `rag ask`, global `--data-dir DIR`; commands
load their code lazily (`--help` 0.7 s, `status` 0.3 s, no torch/FAISS). `rag status` shows
counts, index current/stale, GROBID/LLM up or down, and the next command. `python -m rag.<x>`
still works. End-to-end run in a scratch data dir passed. Found: titles of `ingest files` PDFs
come out wrong (arXiv licence line / venue banner); fix with the second-corpus step. Notes:
`experiments/stage1/cli_step9.md`. 139 tests. Step 10 done (no pre-commit hook: the user
has not decided which repository this goes to; add one later): ruff in `pyproject.toml` (E, F,
W, I, B, UP; line length 120; `experiments/learning/` excluded; prompt files exempt from E501
because wrapping would change the prompt), one mechanical format commit listed in
`.git-blame-ignore-revs`, `zip(..., strict=True)` where lengths must match. Run
`scripts/check.sh` (lint + format check + offline tests, ~2 s; `--fix` applies safe fixes)
before every commit. Stage 1 is complete. Next: collections + second corpus (§0.1 items 2–3),
including the wrong-title fix for `ingest files` PDFs. Do not start Phase 16.

### Phase 7 close-out (2026-09-20)

Full write-up saved to `experiments/learning/phase7_chroma_notes.md` (comparison table + all three
experiment results + corrected conclusion). Summary: default-L2 vs. cosine-space collections
produced identical rankings and, once cosine space was configured, numerically identical scores to
Phase 6 (0.5778 / 0.5121 / 0.4279, matched exactly via `1 - distance`). Root cause: chunk
embeddings are unit-length by construction (`all-MiniLM-L6-v2` bakes a `Normalize` module into its
pipeline), so this is a guarantee of the model, not a property of this dataset — see the corrected
finding under "Chroma vs. manual search" below.

### Environment / machine notes

- **Hardware:** MacBook Pro, Apple Silicon (`arm64`), macOS 26.x.
- **`uv`:** installed via Homebrew (`uv` 0.12.x). Project venv at `.venv/`; Python **3.12** via `uv`.
- **System Python** may be 3.14 — ignore it; always use `uv run …`.
- **Company work** uses Ember/Node (`nvm`) + Java — untouched by `uv`; do not alter those toolchains.
- **Ollama:** installed via `brew install ollama` (0.34.2), running as a `brew services` background
  daemon (`brew services start/stop/restart ollama`, survives reboots). Reachable at
  `http://localhost:11434`. Model `llama3.2:1b` pulled (1.3GB). MPS not available in this sandboxed
  shell environment, so inference runs CPU-only here — a single ~250-word answer took ~16.6s; use
  this as the latency baseline for Phase 13's Top-K/latency experiments, not necessarily
  representative of an unsandboxed run.
- **GROBID:** 0.8.2 cloned and built at `~/AI-project/tools/grobid` (outside the repo: Wapiti
  cannot load models from a path containing a space, like `RAG doc`). Start with
  `cd ~/AI-project/tools/grobid && ./gradlew run --no-daemon` (default JDK 17; JDK 11 also works);
  ready when `curl localhost:8070/api/isalive` returns `true`. ~1.9 GB RAM. Always send
  `consolidateHeader=0`, `consolidateCitations=0`, `consolidateFunders=0` (otherwise it can call
  CrossRef). 0.9.1 needs JDK 21, not installed.
- **Downloads from arXiv:** use a generic User-Agent; never put the user's email in request
  headers.
- **HF Hub:** model downloads work unauthenticated; `HF_TOKEN` warning can be ignored for now.
- **Default embedding model:** `all-MiniLM-L6-v2` (384 dims), local via `sentence-transformers`.
- **Chunk size unit:** word-count proxy for tokens (real tokenizer deferred until needed).
- **`chromadb`** added as a project dependency (`uv add chromadb`, now `1.5.9`) for Phase 7.
  Persistent store lives at `chroma_db/` (created by `PersistentClient(path=...)`) — gitignore this
  if not already. Note: installing/writing into `.venv` needed the sandbox's `all` permission on
  this machine (plain write access was denied) — same likely true for future `uv add`.
- Loading `SentenceTransformer(...)` still makes an outbound HF Hub call (etag/update check) even
  when the model is already cached locally — needed `full_network` sandbox permission to run
  embedding scripts here, not just on first download.

### Key files that exist

| Path | Role |
|---|---|
| `src/rag/ingestion/loader.py` | `LoadedDocument`, `load_txt()` |
| `src/rag/ingestion/chunker.py` | `Chunk`, `chunk_text(..., chunk_size, overlap)` |
| `src/rag/embeddings/embedder.py` | `Embedder`, `EmbeddedChunk` |
| `documents/company_policy.txt` | Short sample (~473 words) — good for smoke tests |
| `documents/employee_handbook.txt` | Longer sample (~2175 words) — use for size/overlap matrices |
| `experiments/learning/run_chunking_experiment.py` | Size × overlap matrix → `chunking_experiment.csv` |
| `experiments/learning/boundary_crossing_check.py` | 0% vs overlap boundary probe |
| `experiments/learning/embedding_similarity_experiment.py` | Phase 5 script (may need creating/running) |
| `experiments/learning/manual_retrieval_debug.py` | Phase 6 — manual cosine top-K debug on `company_policy` |
| `src/rag/retrieval/similarity.py` | Phase 6 — `cosine_similarity()`, `top_k()`, `print_retrieval_debug()` |
| `experiments/learning/chroma_experiment.py` | Phase 7 — same doc/query as Phase 6, via Chroma (default L2 space) |
| `experiments/learning/chroma_cosine_experiment.py` | Phase 7 — same, with `hnsw:space: cosine`; confirms exact score match to Phase 6 |
| `experiments/learning/check_embedding_norms.py` | Phase 7 — confirms `all-MiniLM-L6-v2` embeddings are unit-length (norm always 1.0000) |
| `experiments/learning/phase7_chroma_notes.md` | Phase 7 close-out — full comparison table + findings |
| `chroma_db/` | Phase 7 — persistent Chroma store on disk (created at runtime, gitignored) |
| `src/rag/retrieval/vector_store.py` | Phase 8 — `VectorStore`: thin Chroma wrapper, storage/index layer only |
| `src/rag/retrieval/retriever.py` | Phase 8 — `Retriever`: `index_chunks()` + `retrieve(query, k)`, decoupled from prompts/LLM |
| `experiments/learning/retriever_debug.py` | Phase 8 — end-to-end check; matches Phase 6/7 scores exactly (0.5778/0.5121/0.4279) |
| `src/rag/generation/context_builder.py` | Phase 9 — `build_context()`, `build_prompt()`, `print_context_debug()`; deterministic template, no LLM call |
| `experiments/learning/context_debug.py` | Phase 9 — end-to-end: retrieve(query,k) → exact prompt string printed |
| `src/rag/llm/ollama.py` | Phase 10 — `generate()`, `print_generation_debug()`; direct Ollama HTTP API call, hard-coded on purpose (Phase 11 abstracts it) |
| `experiments/learning/end_to_end_debug.py` | Phase 10 — full pipeline: retrieve → context → Ollama → answer, all stages printed |
| `src/rag/llm/interface.py` | Phase 11 — `LLM` abstract base class; `generate(prompt) -> dict` contract, no provider knowledge |
| `src/rag/llm/ollama.py` | Phase 11 — refactored to `OllamaLLM(LLM)`; `model` now constructor-time config, not a per-call arg |
| `src/rag/llm/groq.py` | Phase 11 — `GroqLLM(LLM)`; hosted free-tier provider, OpenAI-compatible chat completions API |
| `experiments/learning/groq_smoke_test.py` | Phase 11 — isolated `GroqLLM` test, no retrieval/context involved |
| `experiments/learning/groq_list_models.py` | Phase 11 diagnostic — lists Groq's current model catalog (used to fix a decommissioned-model 404) |
| `experiments/learning/end_to_end_groq.py` | Phase 11 — full pipeline with `GroqLLM` swapped in for `OllamaLLM`; proves zero pipeline changes needed |
| `.env` | Real secrets (gitignored): `GROQ_API_KEY`, `GROQ_MODEL=openai/gpt-oss-20b` |
| `experiments/learning/phase12_failure_experiments.py` | Phase 12 — all 3 required experiments in one script (missing-info, retrieval-failure k=1 vs k=3, generation-failure) |
| `experiments/learning/phase12_genfail_ollama.py` | Phase 12 — generation-failure retry with `llama3.2:1b`, same context as the Groq run |
| `experiments/learning/phase12_explore_retrieval_failure.py` / `phase12_confirm_retrieval_failure.py` | Phase 12 — found the retrieval-failure case empirically (8 candidate queries tested, not guessed) |
| `experiments/learning/phase12_failure_notes.md` | Phase 12 close-out — full write-up, all 3 experiments, retrieval-vs-generation diagnosis table |
| `experiments/learning/phase13_topk_experiment.py` | Phase 13 — K∈{1,3,5,10} sweep, 3 queries × 2 docs, retry/backoff for Groq rate limits → `topk_experiment.csv` |
| `experiments/learning/phase13_topk_notes.md` | Phase 13 close-out — full results table + "is more context always better?" finding |
| `experiments/learning/phase14_chunk_topk_matrix.py` | Phase 14 — retrieval grid over chunk size × overlap × K → `phase14_chunk_topk_matrix.csv` |
| `experiments/learning/phase14_generation.py` | Phase 14 — Groq on the 9 cells where needle rank changed → `phase14_generation.csv` |
| `experiments/learning/phase14_chunk_topk_notes.md` | Phase 14 close-out — rank table, generation labels, split-sentence failure |
| `experiments/learning/eval_questions.jsonl` | Phase 15 — 30 questions: direct, multi-step, missing-information, distractor |
| `experiments/learning/check_eval_questions.py` | Phase 15 — each quote must occur once in its document |
| `experiments/learning/phase15_eval_dataset_notes.md` | Phase 15 close-out — record shape and what each question type is for |

Since Stage 1 step 2 (2026-11-17) the code is an installed package: `src/rag/`, imported as
`rag.*` (`from rag.retrieval.retriever import Retriever`). `uv sync` installs it in editable mode,
so no script needs a `sys.path` hack. The learning-track scripts live in `experiments/learning/`
and locate the repo with `project_root = Path(__file__).resolve().parents[2]` (for `documents/`
and `chroma_db/`). `uv run rag` is the CLI entry point (`rag.main:main`).

### Experiment findings so far (do not lose these)

**Chunking matrix** (`employee_handbook`, words as size unit):

- Larger `chunk_size` → fewer chunks (200→11, 500→5, 1000→3 at 0% overlap).
- Overlap increases `total_words` stored (duplication cost); sometimes increases `n_chunks`.
- At 0% overlap, `total_words == original_words` (2175).
- Short docs (`company_policy` ~473 words) make 500/1000 look like “no chunking” — prefer the
  handbook for chunking experiments.

**Boundary crossing** (`chunk_size=200`):

- Probe spanning word 200: with `overlap=0`, left half only in chunk 0, right half only in chunk 1.
- With `overlap=40`, whole probe found in `employee_handbook_1`.
- Concrete proof of why overlap exists.

**Embeddings smoke test:**

- 5 chunks from `company_policy` at size 100 → vectors shape `(384,)`.
- Model download cached after first run.

**Embedding similarity (Phase 5, `all-MiniLM-L6-v2`):**

- A paraphrase (dog sleeping / puppy nap): **0.7135** — high, as expected.
- B unrelated (dog sleeping / stock market): **-0.0002** — near zero / orthogonal.
- C ambiguous Java (language / island): **0.6878** — almost as high as A despite different
  meanings. Lesson: shared surface form + similar sentence frame can fool embeddings; highest
  cosine ≠ always the right sense for retrieval.

**Manual retrieval debug (Phase 6)** — query: “What is the company's refund policy?”
(`company_policy`, chunk_size=100, overlap=10, k=3):

- Rank 1 score **0.5778** — Refund Policy section (correct primary hit).
- Rank 2 score **0.5121** — cancellation text that *references* refunds (related, secondary).
- Rank 3 score **0.4279** — Warranty section (distractor; still somewhat “policy-ish”).
- Lesson: Top-1 can be right while Top-3 still includes near-misses — inspect ranks, don’t trust
  scores blindly.

**Chroma vs. manual search (Phase 7, in progress)** — same doc/chunking/query as the Phase 6 debug
above, `chromadb.PersistentClient` with a **default** (unconfigured) collection:

- Ranking matched Phase 6 exactly: `company_policy_0` (Refund) → `company_policy_5` (cancellation
  referencing refunds) → `company_policy_2` (Warranty). Same chunks, same order.
- Scores did **not** match: Chroma reported L2 distances **0.8444 / 0.9757 / 1.1442** (lower =
  closer) vs. Phase 6 cosine similarities **0.5778 / 0.5121 / 0.4279** (higher = closer) — expected,
  since Chroma's default distance space is squared L2, not cosine.
- **Root cause — CORRECTED after empirical check (2026-09-17):** original hypothesis was that
  `Embedder.embed_texts()` produces non-unit-length vectors since it never passes
  `normalize_embeddings=True`, and that the Phase 7 ranking-match was therefore a coincidental
  property of `company_policy`'s chunks having similar norms. **This was wrong.** Ran
  `experiments/learning/check_embedding_norms.py` on all 6 `company_policy` chunks (word counts 100, 100,
  100, 100, 100, 23) — every single one has `np.linalg.norm(embedding) == 1.0000`, independent of
  chunk length. Traced this to `modules.json` in the cached `all-MiniLM-L6-v2` snapshot: the model
  pipeline is `Transformer → Pooling → 2_Normalize` — a normalization layer is **baked into the
  model itself**, applied regardless of the `normalize_embeddings` flag on `.encode()`. This model
  was packaged specifically for cosine-similarity retrieval.
- **Correct conclusion:** with this specific model, `L2² = 2 − 2·cos_sim` always holds (vectors are
  always unit-length), so default-L2-vs-cosine ranking equivalence is **structurally guaranteed**
  for any dataset embedded with `all-MiniLM-L6-v2` — not a coincidence of `company_policy`,and not
  something a more heterogeneous doc (`employee_handbook`) would break either.
- **What's still true in general:** not every embedding model/pipeline includes a baked-in
  `Normalize` module. If one didn't, magnitude would vary and L2 vs. cosine *could* disagree on
  ranking — the underlying math principle stands, it just isn't a live risk with this model.
- Practical note for later: `metadata={"hnsw:space": "cosine"}` on the Chroma collection (or
  `normalize_embeddings=True` in `Embedder.embed_texts()`, now known to be redundant for this
  model) still matters for *score interpretability* (getting cosine-scale numbers back instead of
  L2 distances) even though it no longer matters for *ranking correctness* here.

**Phase 8 (`retrieve(query, k)`) end-to-end** — same doc/chunking/query as Phases 6/7:

- `Retriever.retrieve()` reproduced scores **exactly** (0.5778 / 0.5121 / 0.4279) via `VectorStore`
  wrapping a cosine-space Chroma collection — confirms the decoupled interface behaves identically
  to the earlier manual/direct-Chroma versions.
- Two real bugs hit and fixed by actually running the code (not caught by eyeballing): (1)
  `vector_store.py` methods `add`/`query` were nested inside `__init__`/inside themselves due to
  indentation, caught via `ast.parse`; (2) `retriever.py` called `vector_store.add(metadatas=...)`
  but `VectorStore.add()`'s parameter is named `metadata` — a keyword-argument name mismatch across
  files, caught only by actually running it (`TypeError: unexpected keyword argument`).

**Phase 9 (context construction) end-to-end** — same query, k=3:

- First run of `context_debug.py` produced a context block with chunk id labels but **no chunk
  text** (`build_context` only interpolated `chunk.chunk_id`, not `chunk.text`, into the f-string) —
  caught by printing the actual prompt string, not by inspecting the code. Fixed by adding
  `{chunk.text}` to the f-string.
- After the fix: prompt = 1642 characters / 255 words for k=3 on `company_policy`.
- Visible, concrete instance of Phase 3's chunk-boundary-crossing problem: the retrieved Warranty
  chunk (`company_policy_2`) also contains the start of the unrelated Privacy Policy section,
  because `chunk_size=100` cut across a section boundary — this is exactly the kind of context
  pollution that later shows up as "why did the answer mention something unrelated" in Phase 10+.
- Also hit a file-location bug: a script was created at `src/evaluation/context_debug.py` instead
  of the top-level `evaluations/context_debug.py`. Since that script computes
  `project_root = Path(__file__).resolve().parent.parent`, being one directory too deep broke both
  the `sys.path` insert and the `documents/` path lookup. Moved the file; no code change needed.
  Lesson: `evaluations/` (top-level) is for experiment scripts; `src/evaluation/` (package) is
  reserved for the real Phase 17 evaluator module, not yet built — don't confuse the two paths.

**Phase 10 (first end-to-end RAG answer)** — `company_policy`, chunk_size=100, overlap=10, k=3,
query "What is the company's refund policy?", model `llama3.2:1b`:

- Full pipeline ran successfully: retrieve → build_prompt → Ollama → answer, all stages printed via
  debug output with nothing hidden.
- Answer was correct and grounded: listed the actual refund policy bullet points (30-day window,
  unused/original packaging, 5-7 day processing, non-refundable digital items, exchange policy,
  cancellation note) — all genuinely present in the retrieved context.
- Notably did **not** leak the unrelated Warranty/Privacy Policy text that was mixed into chunk 3
  (`company_policy_2`) into the answer — one positive data point on grounding, not proof of general
  robustness. Phase 12's mandatory failure experiments (missing-info, retrieval-failure,
  generation-failure) are the actual test of that, not this single run.
- Latency: 16.62s for one ~250-word answer — slow for a 1B model, consistent with CPU-only
  inference (MPS unavailable in this sandboxed environment). Recorded as the Phase 13 baseline.

**Phase 11 (provider abstraction: Ollama + Groq) end-to-end** — same doc/chunking/query as Phase 10:

- `LLM` ABC (`generate(prompt) -> dict`) implemented by both `OllamaLLM` and `GroqLLM`. Swapping
  providers in `end_to_end_groq.py` vs. `end_to_end_debug.py` required changing exactly 2 lines
  (import + construction) — retrieval, chunking, and context construction code is byte-identical.
- Hit a real bug during setup, not caught until actually calling the API: `GROQ_API_KEY`/`GROQ_MODEL`
  in `.env` initially empty (file created but not saved with content — caught via `wc -l .env`
  showing 0 lines), then a 404 from a stale model name (`llama-3.1-8b-instant` — decommissioned,
  not in Groq's current catalog). Diagnosed by listing the live model catalog
  (`experiments/learning/groq_list_models.py`) rather than guessing; fixed by switching to
  `openai/gpt-oss-20b`. Lesson: hosted-provider model names aren't stable long-term — this is a
  real operational difference from local models, where you control the model file directly.
- Latency comparison, same query, same context: `OllamaLLM` (`llama3.2:1b`, local, CPU-only in this
  sandboxed env) 1.91s–16.62s depending on whether the model was already warm; `GroqLLM`
  (`openai/gpt-oss-20b`, hosted, LPU) a consistent 0.73s–1.05s. Both produced correct, grounded
  answers that ignored the unrelated Warranty/Privacy Policy text bleeding into chunk 3 — but Groq's
  answer was noticeably better formatted (markdown headers/bullets) unprompted.
- This is the concrete local-vs-hosted trade-off the project flags for Phase 22: local = private,
  $0 ceiling, latency bound by your own hardware; hosted free-tier = fast/consistent, but your data
  leaves the machine and you're subject to the provider's catalog changes and rate limits.

**Phase 12 (grounding + failure experiments, mandatory)** — full write-up in
`experiments/learning/phase12_failure_notes.md`; summary:

- **Missing information** (`company_policy`, "student/military discounts?"): correctly refused —
  neither retrieval nor generation failure, the document genuinely lacks the info and generation
  didn't fabricate a plausible-sounding policy anyway.
- **Retrieval failure** (`employee_handbook`, "how many late arrivals trigger a review?"): at k=1,
  the correct chunk (score 0.3029) loses to an irrelevant one (0.3139) by a 0.011 margin and is
  excluded entirely — model correctly refused given the wrong context it received (no hallucination,
  but the system still failed). At k=3 with the identical query/model, the correct chunk is
  included and the answer becomes correct. Isolates the failure to retrieval, not generation, since
  the only variable changed was k.
- **Generation failure** (`company_policy`, "how many loyalty points for a $25 discount?" — correct
  answer 500, computed from "100 points per $5" in the retrieved context): `gpt-oss-20b` answered
  correctly; **first attempt at a generation failure didn't materialize** — reported honestly rather
  than forced. Retrying the *exact same retrieved context* against `llama3.2:1b` produced a genuine
  failure: it incorrectly claimed the context "does not mention a discount" despite the rate being
  explicitly stated — isolates the failure to generation (model capacity), not retrieval, since
  context was held constant and only the model changed.
- **Process lesson, not just a result:** the first draft of the experiment script had a real bug —
  forgot to call `index_chunks()` for two of the four runs, producing empty Chroma collections and
  therefore meaningless "I don't know" answers that looked like valid "correctly refused" results.
  Caught only by reading the printed context debug output (blank context) — the same "print the
  exact text sent to the LLM" discipline from Phase 9, this time catching a bug in the *experiment
  harness itself*, not the pipeline under test.
- **General method that made every diagnosis trustworthy:** hold every variable constant except the
  one being tested (k for experiment 2; model for experiment 3), and never conclude anything about
  a "failure" without first checking the actual retrieved chunks/context, not just the final answer.

**Phase 13 (Top-K experiments)** — K∈{1,3,5,10}, full results in
`experiments/learning/phase13_topk_notes.md` / `topk_experiment.csv`; headline finding:

- Retrieval hit rate is monotonically non-decreasing in k, as expected (more chunks can only add
  coverage). Answer correctness is **not** monotonic in k.
- Concrete counterexample: "how many vacation days in the first year" answered correctly at
  k=1/3/5, then flipped to "I'm not sure" at k=10 — with the correct chunk still present
  (`retrieval_hit=True`). Same query, same doc, same model; only k changed, only upward. Root
  cause: `employee_handbook` has many other day-count numbers (30/45/90/etc.) scattered through
  other sections that entered context at k=10 and apparently diluted the model's confidence in the
  correct "15 days" answer it had already stated correctly at smaller k.
- **Answer to "is more context always better?": no**, demonstrated with a controlled example, not
  asserted. Neither extreme is safe by default — k=1 caused a different query (Phase 12's "late
  arrivals" case) to fail from under-retrieval. The right k is query/document-dependent, motivating
  Phase 14/15's move to a real eval set rather than trusting single-query anecdotes.
- k beyond a document's actual chunk count silently clamps (Chroma just returns what exists) rather
  than erroring — `company_policy` (6 chunks total) returns the same 6 chunks for k=5 and k=10.
- Hit Groq's free-tier rate limit (429) mid-sweep from call volume — added retry-with-backoff to
  the script. Concrete, first-hand instance of the "hosted has rate limits, local doesn't" trade-off
  flagged back in Phase 11, this time actually encountered rather than just discussed.
- Generation latency (0.27s–0.74s) didn't scale visibly with context size on Groq's hosted
  inference in this experiment — likely would differ on local CPU inference (Ollama), not tested
  here.

**Phase 14 (chunking × Top-K matrix)** — full write-up in
`experiments/learning/phase14_chunk_topk_notes.md`. Retrieval: 48 rows in
`phase14_chunk_topk_matrix.csv`. Generation: 9 Groq calls in `phase14_generation.csv`.

- `needle_in_index` was true for every row. Rank still moved. Refund days stayed rank 1 at every
  cut. On the handbook, overlap and chunk size help one query and hurt the other: late arrivals
  is rank 1 at 150/0 and rank 2 at the Phase 13 cut 150/15; vacation days is the reverse. At
  chunk size 300, late arrivals falls to rank 6–7 (miss until K=10). Overlap 30 on that cut
  drops vacation days from rank 1 to rank 5.
- Generation followed retrieval on late arrivals. Vacation days at 150/0, K=10 is a generation
  failure: hit is true, answer is "I don't know." The vacation sentence is split across the
  chunk boundary; rank order puts "20 days in years four through seven…" in rank 1 and the
  "15 days … years one through three" fragment at the end of rank 2. The prompt template tells
  the model to refuse when the answer is not clearly in the context.
- The same question at 300/0 keeps that sentence intact in the rank-1 chunk. The model answers
  "15 days" at K=1 and at K=10 (2219 prompt words). A longer prompt was fine once the answering
  sentence was whole and ranked first. The Phase 13 "K=10 hurts" result does not hold for every
  cut.
- No single chunk size, overlap, or K wins both handbook queries. That is why a 20–30 question
  set comes next, rather than picking a default from this grid.

**Phase 15 (evaluation dataset)** — 30 questions in `experiments/learning/eval_questions.jsonl`
(12 direct, 6 multi-step, 6 missing-information, 6 distractor). Notes:
`experiments/learning/phase15_eval_dataset_notes.md`. `check_eval_questions.py` confirms every
`source_quote` and `distractor_quote` occurs exactly once in its document after whitespace
is collapsed.

- Answerable rows carry the span a later Hit@K check should find. Missing-information rows
  (student/military discount, return-shipping payer, same-day delivery, paid holidays,
  referral bonus, equity grants) have a null answer and no quotes.
- One answer is derived, not copied: a $25 loyalty discount is 500 points from "1 point per
  dollar" and "100 points per $5". The number 500 is not in the document.
- Distractors are nearby numbers already known to confuse retrieval and generation: 14-day
  price match vs 30-day refund and 45-day exchange; 20 vacation days vs 15 and 25; P2
  acknowledgment vs the 15-minute P1 rule.

### Collaboration preferences (user-specific)

- User is rebuilding Python muscle memory: **show the full updated function/file** for them to
  **type** (not silent multi-file generation). Boilerplate (deps, long sample docs, `uv add`) may
  be done by the agent when asked.
- Prefer concise explanations; they already know LLM/RAG basics.
- One phase at a time; confirm before jumping ahead.
- When asked “are we done with phase N?”, check this tracker + §7 definition for that phase.

### How to update this tracker

At the end of a productive session (or when the user says “update progress”), the agent must:

1. Flip phase statuses in the table above.
2. Append short findings under “Experiment findings.”
3. Refresh **Next up** to a single concrete next action.
4. Note new files/commands that future sessions will need.

---

## 2. Hard constraints

- **$0 cost.** Default path = local embedding model + local in-memory/Chroma vector search + local
  Ollama LLM. No paid API required at any point in the default path. Hosted providers (Gemini,
  Groq, OpenRouter) are optional, free-tier-only experiments, never required.
- **Hardware:** MacBook Pro, Apple Silicon M4 Pro. Prefer local execution; keep default models
  small enough to run comfortably on this machine.
- **Language:** Python.
- **Env management:** `uv`. Use a `uv`-managed virtual environment; don't hand-roll `pip`/`venv`.
- **Default LLM runtime:** Ollama, accessed over its local API. The app must never hard-code a
  specific model name into pipeline logic — model choice is configuration.

## 3. Non-negotiable architecture principles

- **The RAG pipeline depends on an LLM interface, not a provider.** `LLM_PROVIDER=ollama` today,
  `LLM_PROVIDER=gemini` tomorrow, with zero changes to retrieval/chunking/context code.
- **Retrieval is independent from the LLM layer.** `retrieve(query, k)` returns chunk + score +
  metadata + rank and knows nothing about prompts or providers.
- **Every stage must have a debug/inspection mode**: extracted text stats, chunk metadata, raw
  embeddings, similarity scores + ranks, the exact context string sent to the LLM, and per-query
  observability (timings, token counts, provider/model used).
- **Manual before magic.** Implement cosine similarity + NumPy top-K search by hand *before*
  introducing Chroma. Introduce Chroma only after the user understands what it replaces.
- **No premature abstraction.** Don't build things "for architecture's sake." Add a seam only when
  a concrete next step needs it (e.g. add the LLM interface only once Ollama already works).

## 4. What NOT to use initially

Do not introduce these until the manual/local foundation works and the user explicitly wants to
explore them: LangChain, LlamaIndex, Pinecone, Weaviate, agents, MCP, multi-agent systems,
fine-tuning, custom transformer implementations. If a framework is introduced later, it must be to
answer "what problem does this framework solve?" — not as a shortcut.

## 5. Tech stack

| Concern | Choice |
|---|---|
| Language / env | Python via `uv` |
| Local LLM | Ollama (small model to start) |
| Optional hosted LLMs | Gemini, Groq, OpenRouter (free tier only) |
| Local embeddings | `sentence-transformers` |
| Manual vector search | NumPy (cosine similarity, top-K) |
| Vector DB (later) | Chroma, local/persistent |
| Doc formats | `.txt` first, then `.md`, then `.pdf` |

## 6. Repository structure

Superseded by `docs/production_plan.md` §3 (package is now `src/rag/`). Kept for history.

```text
rag-document-qa/
├── src/
│   ├── ingestion/{loader.py, chunker.py}
│   ├── embeddings/embedder.py
│   ├── retrieval/{similarity.py, vector_store.py, retriever.py}
│   ├── llm/{interface.py, ollama.py, ...}
│   ├── generation/answerer.py
│   ├── evaluation/evaluator.py
│   ├── config/config.py
│   └── main.py
├── tests/
├── documents/
├── evaluations/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

Structure may evolve — don't force it if a phase reveals a better shape, but don't add
speculative modules ahead of need.

---

## 7. Build order (phases)

**Paused as of 2026-10-10.** Phases 1–15 are done and are the foundation. Do not continue this
list at Phase 16. The active work is the production-grade RAG described in §1 / §1b. Use a phase
below only if the user reopens it, or if a production slice needs the same measurement.

Work through these **in order, one phase at a time**. Do not skip ahead or pre-build later phases.

1. **Ingestion** — load `.txt`, extract text, report char/word counts. Explain why ingestion is
   separate from chunking, why extraction quality matters, why PDFs are hard.
2. **Chunking** — fixed-size chunker; every chunk carries `document_id`, `chunk_id`, `position`,
   `text` (later `page` for PDFs).
3. **Chunking experiments** — vary chunk size (200/500/1000 tokens) × overlap (0/10/20%). Measure
   retrieval quality, context size, answer quality, latency, tokens. Investigate boundary-crossing
   information loss.
4. **Local embeddings** — embed chunks with `sentence-transformers`; store chunk+embedding+metadata.
5. **Embedding experiments** — run the "sleeping dog / puppy nap", "dog sleeping / stock market",
   and "Java language / Java island" comparisons to build intuition for when embeddings work and
   when they don't.
6. **Manual vector search** — cosine similarity + NumPy top-K by hand, with rank/score/text
   inspection (retrieval debug mode).
7. **Chroma** — introduce only after step 6 works. Explicitly compare "what we built ourselves" vs
   "what Chroma now does for us" (persistence, indexing, metadata, scaling).
8. **Query retrieval** — `retrieve(query, k)` → chunk/score/metadata/rank, decoupled from the LLM.
9. **Context construction** — deterministic template combining question + retrieved chunks, with a
   debug mode that prints the exact text sent to the LLM.
10. **Local LLM (Ollama)** — wire retrieval → context → Ollama → answer, small model first.
11. **Provider abstraction** — introduce the `LLM` interface only after Ollama works end-to-end;
    plug in one hosted provider (Gemini/Groq/OpenRouter) using its free tier.
12. **Grounding + failure experiments (mandatory)**:
    - Missing-information questions → classify "correctly refused" vs "hallucinated".
    - Retrieval-failure case → answer exists, inspect whether the right chunk was retrieved.
    - Generation-failure case → correct chunk retrieved, LLM still gets it wrong.
    - Explicitly distinguish retrieval failure from generation failure every time.
13. **Top-K experiments** — K ∈ {1,3,5,10}; measure relevance, context size, answer quality,
    latency, tokens. Is more context always better?
14. **Chunking × Top-K experiment matrix** — run the eval set across a grid of (chunk size,
    overlap, K) combinations and record results.
15. **Evaluation dataset** — 20–30 questions: direct, multi-step, missing-information, distractor.
16. **Retrieval evaluation** — Hit@K / Recall@K against expected sources.
17. **Generation evaluation** — correctness/relevance/groundedness/unsupported-claims, manual first,
    LLM-as-judge only after manual scoring is understood.
18. **Metadata filtering** — add `document_id/filename/page/section/chunk_id`; filter + semantic
    search together.
19. **Stretch: hybrid search** (keyword + semantic) — motivate with exact-identifier queries.
20. **Stretch: reranking** — vector search top-20 → reranker → top-5 → LLM.
21. **Observability** — log per query: embedding/retrieval/generation time, top_k, retrieved chunks,
    scores, input/output/total tokens, model, provider.
22. **Local vs hosted comparison** — hold documents/chunking/embeddings/store/retrieval/eval set
    constant, swap only the LLM, compare quality/hallucination/latency/tokens/failure behavior.
23. **Tests** — chunking, similarity ranking, retrieval Top-K correctness, context construction,
    config validation (missing/invalid provider), provider-abstraction interchangeability.
24. **README** — document architecture and experiment results.
25. **Final knowledge test** (see §9) before declaring the project done.

## 8. How the agent should behave while working on this project

- **Check §1b Progress tracker first** every session. Resume at **Next up**; do not redo completed
  phases or re-ask setup questions already answered there.
- **Mentor, don't hand off.** Act like a senior Applied AI Engineer pairing with an experienced
  developer, not a code-generation service. Bias toward the user typing/reviewing code themselves.
- **Show full code for the user to type** when implementing a step (user preference — rebuilding
  Python familiarity). Prefer one complete file/function over scattered snippets. Agent may still
  install deps / add long sample documents when the user asks for boilerplate help.
- **One step at a time.** At the start of a new phase, briefly explain the concept/trade-off, give
  the *next concrete step only*, and let the user implement or explicitly ask you to. Don't
  silently build multiple phases ahead of where the user currently is.
- **Never skip the debug/inspection tooling** for a phase to "save time" — the visibility is the
  point of the project.
- **Record experiments, don't just assert.** Whenever a phase includes an experiment, produce an
  artifact (table, log file, note in `experiments/learning/`) with actual numbers — never conclude with
  "it works" without a measurement. Also copy key numbers into §1b findings.
- **When something fails (bad retrieval, hallucination, wrong answer), don't immediately patch it.**
  First diagnose: is this a retrieval failure or a generation failure? Only fix after the user
  understands why it happened.
- **Don't reach for a bigger model, more chunks, or a framework as a first response to a bad
  result.** Treat every "fix" as a hypothesis to test, not an assumption.
- **Respect the cost constraint.** Never introduce a paid API call as a required path.
- **Update §1b** when the user asks to update progress, or at natural phase boundaries before
  ending a session.

## 8b. Learning-mode behavior

When the user is using the application to learn from a book/blog/document:

- **Source first.** Retrieve relevant source material before making factual claims about the
  material.
- **Cite the source.** Explanations should identify the relevant document/chapter/page/section/
  chunk when available.
- **Do not over-answer.** If the user asks to be quizzed, ask the question and wait for their
  answer.
- **Do not immediately reveal answers.** In quiz/Socratic mode, let the learner attempt first
  unless they explicitly ask for the answer.
- **Teach reasoning, not memorization.** Prefer questions that test application, comparison,
  prediction, and explanation.
- **Challenge the learner.** When an answer is partially correct, identify the correct part and
  probe the incorrect assumption.
- **Separate source from interpretation.** Say when something is directly supported by the text
  versus an explanatory inference.
- **Do not manufacture mastery.** A correct answer once is not proof of durable understanding.
- **Preserve reading context.** If the user is studying a specific chapter/section, prefer that
  context unless the task explicitly requires the wider corpus.
- **Treat learner answers as data, not truth.** Compare them against retrieved source evidence
  before grading.
- **Evaluate the tutor itself.** Learning interactions need tests for correctness, groundedness,
  grading consistency, and unsupported feedback.

## 9. Final knowledge test (do not skip)

Before marking the project complete, quiz the user on the following — ask the questions, **do not
answer them yourself**, and let them attempt answers first:

**Embeddings:** (1) What does an embedding represent? (2) Why can semantically similar text
produce similar vectors? (3) What are the limitations of embeddings?

**Chunking:** (4) Why do we chunk documents? (5) What happens with very small chunks? (6) What
happens with very large chunks? (7) Why does overlap exist?

**Retrieval:** (8) Why do we embed the query? (9) What does Top-K mean? (10) What does cosine
similarity tell us? (11) Why can the highest similarity score still correspond to an irrelevant
chunk?

**RAG:** (12) What makes a system RAG? (13) Where does retrieval happen? (14) Where does
generation happen? (15) What exact information is sent to the LLM?

**Failures:** (16) How do you distinguish retrieval failure from generation failure? (17) Can an
LLM hallucinate even when the correct chunk is retrieved? (18) Why isn't increasing Top-K always
the solution?

**Evaluation:** (19) How would you measure retrieval quality? (20) How would you measure answer
quality? (21) Why isn't testing five questions enough?

**Architecture:** (22) Why shouldn't the RAG pipeline depend directly on Ollama? (23) What does
the LLM provider abstraction give us? (24) Why might you choose a local model over a hosted model?
(25) Why might you choose a hosted model over a local model?

## 10. Definition of done

See the full checklist the user maintains for this project (project setup via `uv`; Ollama
working; ingestion → chunking → embeddings → manual search → Chroma → retrieval → context →
generation, each inspectable; provider abstraction with one hosted provider pluggable; learning
modes (explain/quiz/Socratic) grounded in retrieved source material;
missing-info/retrieval-failure/generation-failure experiments done; chunk-size, overlap, and Top-K
experiments done; evaluation dataset + retrieval metrics + generation evaluation; observability;
tests; README with architecture and experiment write-ups; and the user can explain the entire
pipeline unaided). The project isn't "done" until the knowledge test in §9 has been given.

## 11. Where this leads

This pipeline is the foundation for later systems: RAG → Tool Calling → Agents → Agent Runtime →
Evaluation → Production AI Systems. Keep that framing in mind when explaining *why* a given
architectural discipline (interfaces, observability, evaluation-first thinking) matters beyond
this one project.
