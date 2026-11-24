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
rebuild. Phases 16–25 stay in §7 as a backlog that can be revisited after that work, or
folded into it when a production slice needs the same measurement.

The standing reminder to explain the Phase 16 retrieval results in plain language before anything
else is **cancelled**.

---

## 1b. Current state (keep this short)

> Agents: read this, then `HANDOFF.md` and `docs/production_plan.md` §0.2 (the ordered roadmap).
> Don't add a per-step history here or a write-up file per step (user's rule, 2026-11-24): keep
> results that serve the dev — a line in `docs/results.md` for numbers the README will cite, the
> script/CSV that produced them in `experiments/`, and an updated "Next up" below. Git log has the rest.

**Learning track (Phases 1–15): closed.** Phase 16 has a run (`experiments/learning/phase16_*`)
but no notes; don't resume it unless the user asks.

**Product:** a generic, production-grade RAG for a single Mac, released as a packaged localhost app
(point at a folder or upload files; models download on first run; Ollama is a separate install).
Research papers were the first dataset; the app must not be paper-specific.

**Done (roadmap steps 1–25 and 30; 23 skipped for now):** ingest (PyMuPDF layout parser by default, GROBID
optional; pdf/docx/html/txt/md; recursive folders), chunker v2-350, bge-small embeddings + exact
FAISS, hand-written BM25, RRF hybrid, cross-encoder reranker (fused default for `ask`), cited
answers with guardrails and source sanitising, eval sets + `rag eval` + a zero-tolerance quality
gate, FastAPI server + web UI (ask, search, documents/sync, activity, evaluation), per-request logs,
collections with folder sync and uploads. Numbers: `docs/results.md`.

**Next up:** step 25 (index benchmark) is done: the app stays on exact FAISS search (8 ms at 1M chunks, lossless; HNSW loses recall under the collection filter; SQ8 is the fallback if memory runs short). Numbers in `docs/results.md`, script `experiments/stage4/index_bench.py`. Next: step 26 (reranker at scale) was pitched 2026-11-24; waiting for the user's answers (see `HANDOFF.md`). Then 27, 28 (packaging), 29, 31 (README), 32 (knowledge test); 23 optional.

**Open decisions / known gaps:** PyMuPDF is AGPL (decide before step 28); human review of the
gov-docs questions and the step-16 answer labels is pending; prompt v3 says "paper" once (needs a v4
+ answer eval); browser folder uploads flatten subfolder names; the embedding cache never prunes;
search stalls during a sync (GIL); Ollama prompt reading is slow under memory pressure.

### Rules the user set (keep to these)

- **Pitch every step before building** (problem, build, why, proof, risks, out of scope, decisions
  with a recommendation); wait for "go".
- **Ask before adding any dependency, library, model, service or download**, even one the plan
  names. Downloads: file, source, size first.
- Claude writes the code (the user delegated it); explain the reasoning so the user learns.
- Run `scripts/check.sh` (lint, format, offline tests, quality gate) before every commit. If the
  gate fails on purpose: `rag eval gate --update "reason"` in the same commit. Commits stay local;
  never push.
- Diagnose before fixing (retrieval vs generation); measure, don't assert.
- Stop Ollama (`brew services stop ollama`) and any `rag serve` when done.
- Untrusted text in the UI only as text nodes (a test greps for `innerHTML`).

### Environment / machine notes

- MacBook Pro, Apple M4 Pro, 24 GB, macOS 26.x. `uv` (Homebrew) with Python 3.12 in `.venv/`;
  always `uv run …`. Company toolchains (Node via nvm, Java) are untouched.
- Ollama via `brew services start|stop ollama`, model `qwen2.5:7b-instruct`, `num_ctx` 8192.
- GROBID 0.8.2 (optional) at `~/AI-project/tools/grobid` (a space in the path breaks it):
  `cd ~/AI-project/tools/grobid && ./gradlew run --no-daemon`; always send consolidation off.
- arXiv and other downloads: generic User-Agent, never the user's email.
- `faiss` and `torch` load fine together; `chromadb` + `torch` in one process crash on a duplicate
  OpenMP runtime — run Chroma in its own process.
- Data in `data/` (gitignored): `rag.db` (schema v6), indexes, eval runs/answers, logs. Commands:
  `uv run rag status | serve | sync | ask | search | logs | eval …`. Tests: `uv run pytest -q`.
- `.claude/launch.json` has a `rag-serve` preview config for the in-app browser.

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

- **Check §1b and `HANDOFF.md` first** every session. Resume at **Next up**; do not redo completed
  steps or re-ask setup questions already answered there.
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
  artifact (the script and its CSV/JSON output in `experiments/`) with actual numbers — never conclude
  with "it works" without a measurement. Numbers the README will cite get one line in
  `docs/results.md`; no write-up file per step.
- **When something fails (bad retrieval, hallucination, wrong answer), don't immediately patch it.**
  First diagnose: is this a retrieval failure or a generation failure? Only fix after the user
  understands why it happened.
- **Don't reach for a bigger model, more chunks, or a framework as a first response to a bad
  result.** Treat every "fix" as a hypothesis to test, not an assumption.
- **Respect the cost constraint.** Never introduce a paid API call as a required path.
- **Update §1b "Next up"** (a few lines, not a history) when a step finishes or the user asks.

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
