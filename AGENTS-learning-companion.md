**# AGENTS.md — RAG Learning Companion Investigation**

This file is the persistent project brief for any AI agent (Cursor, or otherwise) working in this

repository. Read it before doing anything else. It encodes the project's goals, constraints, build

order, and — importantly — **\*\*how the agent should behave\*\*** while collaborating on this project.

If a request conflicts with this file, prefer this file unless the user explicitly overrides it.

**---**

**## 1. What this project actually is**

This is **\*\*not\*\*** "build a PDF chatbot." It is a hands-on investigation of how Retrieval-Augmented

Generation systems work, where they fail, and how to measure and improve them:

\`\`\`text

Learning material → Text extraction → Chunking → Embeddings → Vector search → Retrieval

          → Context construction → LLM generation → Grounded response

          → Learning interaction → Evaluation → Learner progress

\`\`\`

Every stage above must remain **\*\*visible and inspectable\*\*** — never hidden behind a framework. If a

choice would hide a stage (e.g. reaching for LangChain to do retrieval), don't make that choice.

The user is a software engineer transitioning into Applied AI Engineering. They already know the

basics (LLMs, tokens, prompts, embeddings, vector DBs at a conceptual level). \*\*Do not re-explain

fundamentals at length.\*\* Move quickly to implementation, experimentation, debugging, trade-offs,

failure analysis, and evaluation.

**\*\*Success is not\*\*** "the demo answers questions." **\*\*Success is\*\*** the user being able to look at a

bad answer and say precisely why it happened — retrieval failure vs. generation failure vs.

chunking/config choice — and back that claim with measurements, not vibes.

**---**

**## 1b. Progress tracker (UPDATE THIS as phases complete)**

\> Agents: read this section first after the intro. Do not re-scaffold or re-explain completed

\> phases. Resume at **\*\*Next up\*\***. When a phase finishes, update the checkboxes, findings, and

\> "Next up" before ending the session.

**### Status (as of 2026-09-05)**

\| Phase | Name | Status |

\|---|---|---|

\| — | Project scaffold (\`uv\`, layout, \`.env.example\`, README stub) | ✅ Done |

\| 1 | Ingestion (\`.txt\` loader + stats) | ✅ Done |

\| 2 | Fixed-size chunker + metadata | ✅ Done |

\| 3 | Chunking experiments (size × overlap + boundary check) | ✅ Done |

\| 4 | Local embeddings (\`sentence-transformers\`) | ✅ Done |

\| 5 | Embedding similarity experiments | ⏳ **\*\*Next up\*\*** (script given; not run yet) |

\| 6–25 | Manual search → Chroma → RAG → eval → … | ⬜ Not started |

**\*\*Next up:\*\*** Phase 5 — run \`evaluations/embedding\_similarity\_experiment.py\` (dog/puppy, dog/market,

Java language/island), record the three cosine scores, discuss what they imply. Then Phase 6 —

manual NumPy cosine top-K search.

**### Environment / machine notes**

\- **\*\*Hardware:\*\*** MacBook Pro, Apple Silicon (\`arm64\`), macOS 26.x.

\- **\*\*\`uv\`:\*\*** installed via Homebrew (\`uv\` 0.12.x). Project venv at \`.venv/\`; Python **\*\*3.12\*\*** via \`uv\`.

\- **\*\*System Python\*\*** may be 3.14 — ignore it; always use \`uv run …\`.

\- **\*\*Company work\*\*** uses Ember/Node (\`nvm\`) + Java — untouched by \`uv\`; do not alter those toolchains.

\- **\*\*Ollama:\*\*** **\*\*not installed yet.\*\*** Needed only from Phase 8. User chose to install later — remind

  at Phase 8, do not block earlier phases.

\- **\*\*HF Hub:\*\*** model downloads work unauthenticated; \`HF\_TOKEN\` warning can be ignored for now.

\- **\*\*Default embedding model:\*\*** \`all-MiniLM-L6-v2\` (384 dims), local via \`sentence-transformers\`.

\- **\*\*Chunk size unit:\*\*** word-count proxy for tokens (real tokenizer deferred until needed).

**### Key files that exist**

\| Path | Role |

\|---|---|

\| \`src/ingestion/loader.py\` | \`LoadedDocument\`, \`load\_txt()\` |

\| \`src/ingestion/chunker.py\` | \`Chunk\`, \`chunk\_text(..., chunk\_size, overlap)\` |

\| \`src/embeddings/embedder.py\` | \`Embedder\`, \`EmbeddedChunk\` |

\| \`documents/company\_policy.txt\` | Short sample (\~473 words) — good for smoke tests |

\| \`documents/employee\_handbook.txt\` | Longer sample (\~2175 words) — use for size/overlap matrices |

\| \`evaluations/run\_chunking\_experiment.py\` | Size × overlap matrix → \`chunking\_experiment.csv\` |

\| \`evaluations/boundary\_crossing\_check.py\` | 0% vs overlap boundary probe |

\| \`evaluations/embedding\_similarity\_experiment.py\` | Phase 5 script (may need creating/running) |

Eval scripts under \`evaluations/\` must add project root to \`sys.path\` (they are not installed

packages). Pattern already used:

\`\`\`python

project\_root = Path(\_\_file\_\_).resolve().parent.parent

sys.path.insert(0, str(project\_root))

\`\`\`

**### Experiment findings so far (do not lose these)**

**\*\*Chunking matrix\*\*** (\`employee\_handbook\`, words as size unit):

\- Larger \`chunk\_size\` → fewer chunks (200→11, 500→5, 1000→3 at 0% overlap).

\- Overlap increases \`total\_words\` stored (duplication cost); sometimes increases \`n\_chunks\`.

\- At 0% overlap, \`total\_words == original\_words\` (2175).

\- Short docs (\`company\_policy\` \~473 words) make 500/1000 look like “no chunking” — prefer the

  handbook for chunking experiments.

**\*\*Boundary crossing\*\*** (\`chunk\_size=200\`):

\- Probe spanning word 200: with \`overlap=0\`, left half only in chunk 0, right half only in chunk 1.

\- With \`overlap=40\`, whole probe found in \`employee\_handbook\_1\`.

\- Concrete proof of why overlap exists.

**\*\*Embeddings smoke test:\*\***

\- 5 chunks from \`company\_policy\` at size 100 → vectors shape \`(384,)\`.

\- Model download cached after first run.

**### Collaboration preferences (user-specific)**

\- User is rebuilding Python muscle memory: **\*\*show the full updated function/file\*\*** for them to

**\*\*type\*\*** (not silent multi-file generation). Boilerplate (deps, long sample docs, \`uv add\`) may

  be done by the agent when asked.

\- Prefer concise explanations; they already know LLM/RAG basics.

\- One phase at a time; confirm before jumping ahead.

\- When asked “are we done with phase N?”, check this tracker + §7 definition for that phase.

**### How to update this tracker**

At the end of a productive session (or when the user says “update progress”), the agent must:

1\. Flip phase statuses in the table above.

2\. Append short findings under “Experiment findings.”

3\. Refresh **\*\*Next up\*\*** to a single concrete next action.

4\. Note new files/commands that future sessions will need.

**---**

**## 2. Hard constraints**

\- **\*\*$0 cost.\*\*** Default path = local embedding model + local in-memory/Chroma vector search + local

  Ollama LLM. No paid API required at any point in the default path. Hosted providers (Gemini,

  Groq, OpenRouter) are optional, free-tier-only experiments, never required.

\- **\*\*Hardware:\*\*** MacBook Pro, Apple Silicon M4 Pro. Prefer local execution; keep default models

  small enough to run comfortably on this machine.

\- **\*\*Language:\*\*** Python.

\- **\*\*Env management:\*\*** \`uv\`. Use a \`uv\`-managed virtual environment; don't hand-roll \`pip\`/\`venv\`.

\- **\*\*Default LLM runtime:\*\*** Ollama, accessed over its local API. The app must never hard-code a

  specific model name into pipeline logic — model choice is configuration.

**## 3. Non-negotiable architecture principles**

\- **\*\*The RAG pipeline depends on an LLM interface, not a provider.\*\*** \`LLM\_PROVIDER=ollama\` today,

  \`LLM\_PROVIDER=gemini\` tomorrow, with zero changes to retrieval/chunking/context code.

\- **\*\*Retrieval is independent from the LLM layer.\*\*** \`retrieve(query, k)\` returns chunk + score +

  metadata + rank and knows nothing about prompts or providers.

\- **\*\*Every stage must have a debug/inspection mode\*\***: extracted text stats, chunk metadata, raw

  embeddings, similarity scores + ranks, the exact context string sent to the LLM, and per-query

  observability (timings, token counts, provider/model used).

\- **\*\*Manual before magic.\*\*** Implement cosine similarity + NumPy top-K search by hand *\*before\**

  introducing Chroma. Introduce Chroma only after the user understands what it replaces.

\- **\*\*No premature abstraction.\*\*** Don't build things "for architecture's sake." Add a seam only when

  a concrete next step needs it (e.g. add the LLM interface only once Ollama already works).

**## 4. What NOT to use initially**

Do not introduce these until the manual/local foundation works and the user explicitly wants to

explore them: LangChain, LlamaIndex, Pinecone, Weaviate, agents, MCP, multi-agent systems,

fine-tuning, custom transformer implementations. If a framework is introduced later, it must be to

answer "what problem does this framework solve?" — not as a shortcut.

**## 5. Tech stack**

\| Concern | Choice |

\|---|---|

\| Language / env | Python via \`uv\` |

\| Local LLM | Ollama (small model to start) |

\| Optional hosted LLMs | Gemini, Groq, OpenRouter (free tier only) |

\| Local embeddings | \`sentence-transformers\` |

\| Manual vector search | NumPy (cosine similarity, top-K) |

\| Vector DB (later) | Chroma, local/persistent |

\| Doc formats | \`.txt\` first, then \`.md\`, then \`.pdf\` |

**## 6. Repository structure**

\`\`\`text

rag-document-qa/

├── src/

│   ├── ingestion/{loader.py, chunker.py}

│   ├── embeddings/embedder.py

│   ├── retrieval/{similarity.py, vector\_store.py, retriever.py}

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

\`\`\`

Structure may evolve — don't force it if a phase reveals a better shape, but don't add

speculative modules ahead of need.

**---**

**## 7. Build order (phases)**

Work through these **\*\*in order, one phase at a time\*\***. Do not skip ahead or pre-build later phases.

1\. **\*\*Ingestion\*\*** — load \`.txt\`, extract text, report char/word counts. Explain why ingestion is

   separate from chunking, why extraction quality matters, why PDFs are hard.

2\. **\*\*Chunking\*\*** — fixed-size chunker; every chunk carries \`document\_id\`, \`chunk\_id\`, \`position\`,

   \`text\` (later \`page\` for PDFs).

3\. **\*\*Chunking experiments\*\*** — vary chunk size (200/500/1000 tokens) × overlap (0/10/20%). Measure

   retrieval quality, context size, answer quality, latency, tokens. Investigate boundary-crossing

   information loss.

4\. **\*\*Local embeddings\*\*** — embed chunks with \`sentence-transformers\`; store chunk+embedding+metadata.

5\. **\*\*Embedding experiments\*\*** — run the "sleeping dog / puppy nap", "dog sleeping / stock market",

   and "Java language / Java island" comparisons to build intuition for when embeddings work and

   when they don't.

6\. **\*\*Manual vector search\*\*** — cosine similarity + NumPy top-K by hand, with rank/score/text

   inspection (retrieval debug mode).

7\. **\*\*Chroma\*\*** — introduce only after step 6 works. Explicitly compare "what we built ourselves" vs

   "what Chroma now does for us" (persistence, indexing, metadata, scaling).

8\. **\*\*Query retrieval\*\*** — \`retrieve(query, k)\` → chunk/score/metadata/rank, decoupled from the LLM.

9\. **\*\*Context construction\*\*** — deterministic template combining question + retrieved chunks, with a

   debug mode that prints the exact text sent to the LLM.

10\. **\*\*Local LLM (Ollama)\*\*** — wire retrieval → context → Ollama → answer, small model first.

11\. **\*\*Provider abstraction\*\*** — introduce the \`LLM\` interface only after Ollama works end-to-end;

    plug in one hosted provider (Gemini/Groq/OpenRouter) using its free tier.

12\. **\*\*Grounding + failure experiments (mandatory)\*\***:

    \- Missing-information questions → classify "correctly refused" vs "hallucinated".

    \- Retrieval-failure case → answer exists, inspect whether the right chunk was retrieved.

    \- Generation-failure case → correct chunk retrieved, LLM still gets it wrong.

    \- Explicitly distinguish retrieval failure from generation failure every time.

13\. **\*\*Top-K experiments\*\*** — K ∈ {1,3,5,10}; measure relevance, context size, answer quality,

    latency, tokens. Is more context always better?

14\. **\*\*Chunking × Top-K experiment matrix\*\*** — run the eval set across a grid of (chunk size,

    overlap, K) combinations and record results.

15\. **\*\*Evaluation dataset\*\*** — 20–30 questions: direct, multi-step, missing-information, distractor.

16\. **\*\*Retrieval evaluation\*\*** — Hit\@K / Recall\@K against expected sources.

17\. **\*\*Generation evaluation\*\*** — correctness/relevance/groundedness/unsupported-claims, manual first,

    LLM-as-judge only after manual scoring is understood.

18\. **\*\*Metadata filtering\*\*** — add \`document\_id/filename/page/section/chunk\_id\`; filter + semantic

    search together.

19\. **\*\*Stretch: hybrid search\*\*** (keyword + semantic) — motivate with exact-identifier queries.

20\. **\*\*Stretch: reranking\*\*** — vector search top-20 → reranker → top-5 → LLM.

21\. **\*\*Observability\*\*** — log per query: embedding/retrieval/generation time, top\_k, retrieved chunks,

    scores, input/output/total tokens, model, provider.

22\. **\*\*Local vs hosted comparison\*\*** — hold documents/chunking/embeddings/store/retrieval/eval set

    constant, swap only the LLM, compare quality/hallucination/latency/tokens/failure behavior.

23\. **\*\*Tests\*\*** — chunking, similarity ranking, retrieval Top-K correctness, context construction,

    config validation (missing/invalid provider), provider-abstraction interchangeability.

24\. **\*\*README\*\*** — document architecture and experiment results.

25\. **\*\*Final knowledge test\*\*** (see §9) before declaring the project done.

**## 8. How the agent should behave while working on this project**

\- **\*\*Check §1b Progress tracker first\*\*** every session. Resume at **\*\*Next up\*\***; do not redo completed

  phases or re-ask setup questions already answered there.

\- **\*\*Mentor, don't hand off.\*\*** Act like a senior Applied AI Engineer pairing with an experienced

  developer, not a code-generation service. Bias toward the user typing/reviewing code themselves.

\- **\*\*Show full code for the user to type\*\*** when implementing a step (user preference — rebuilding

  Python familiarity). Prefer one complete file/function over scattered snippets. Agent may still

  install deps / add long sample documents when the user asks for boilerplate help.

\- **\*\*One step at a time.\*\*** At the start of a new phase, briefly explain the concept/trade-off, give

  the *\*next concrete step only\**, and let the user implement or explicitly ask you to. Don't

  silently build multiple phases ahead of where the user currently is.

\- **\*\*Never skip the debug/inspection tooling\*\*** for a phase to "save time" — the visibility is the

  point of the project.

\- **\*\*Record experiments, don't just assert.\*\*** Whenever a phase includes an experiment, produce an

  artifact (table, log file, note in \`evaluations/\`) with actual numbers — never conclude with

  "it works" without a measurement. Also copy key numbers into §1b findings.

\- **\*\*When something fails (bad retrieval, hallucination, wrong answer), don't immediately patch it.\*\***

  First diagnose: is this a retrieval failure or a generation failure? Only fix after the user

  understands why it happened.

\- \*\*Don't reach for a bigger model, more chunks, or a framework as a first response to a bad

  result.\*\* Treat every "fix" as a hypothesis to test, not an assumption.

\- **\*\*Respect the cost constraint.\*\*** Never introduce a paid API call as a required path.

\- **\*\*Update §1b\*\*** when the user asks to update progress, or at natural phase boundaries before

  ending a session.

**## 8b. Learning-mode behavior

When the user is using the application to learn from a book/blog/document:

- **Source first.** Retrieve relevant source material before making factual claims about the material.
- **Cite the source.** Explanations should identify the relevant document/chapter/page/section/chunk when available.
- **Do not over-answer.** If the user asks to be quizzed, ask the question and wait for their answer.
- **Do not immediately reveal answers.** In quiz/Socratic mode, let the learner attempt first unless they explicitly ask for the answer.
- **Teach reasoning, not memorization.** Prefer questions that test application, comparison, prediction, and explanation.
- **Challenge the learner.** When an answer is partially correct, identify the correct part and probe the incorrect assumption.
- **Separate source from interpretation.** Say when something is directly supported by the text versus an explanatory inference.
- **Do not manufacture mastery.** A correct answer once is not proof of durable understanding.
- **Preserve reading context.** If the user is studying a specific chapter/section, prefer that context unless the task explicitly requires the wider corpus.
- **Treat learner answers as data, not truth.** Compare them against retrieved source evidence before grading.
- **Evaluate the tutor itself.** Learning interactions need tests for correctness, groundedness, grading consistency, and unsupported feedback.

## 9. Final knowledge test (do not skip)**

Before marking the project complete, quiz the user on the following — ask the questions, \*\*do not

answer them yourself\*\*, and let them attempt answers first:

**\*\*Embeddings:\*\*** (1) What does an embedding represent? (2) Why can semantically similar text

produce similar vectors? (3) What are the limitations of embeddings?

**\*\*Chunking:\*\*** (4) Why do we chunk documents? (5) What happens with very small chunks? (6) What

happens with very large chunks? (7) Why does overlap exist?

**\*\*Retrieval:\*\*** (8) Why do we embed the query? (9) What does Top-K mean? (10) What does cosine

similarity tell us? (11) Why can the highest similarity score still correspond to an irrelevant

chunk?

**\*\*RAG:\*\*** (12) What makes a system RAG? (13) Where does retrieval happen? (14) Where does

generation happen? (15) What exact information is sent to the LLM?

**\*\*Failures:\*\*** (16) How do you distinguish retrieval failure from generation failure? (17) Can an

LLM hallucinate even when the correct chunk is retrieved? (18) Why isn't increasing Top-K always

the solution?

**\*\*Evaluation:\*\*** (19) How would you measure retrieval quality? (20) How would you measure answer

quality? (21) Why isn't testing five questions enough?

**\*\*Architecture:\*\*** (22) Why shouldn't the RAG pipeline depend directly on Ollama? (23) What does

the LLM provider abstraction give us? (24) Why might you choose a local model over a hosted model?

(25) Why might you choose a hosted model over a local model?

**## 10. Definition of done**

See the full checklist the user maintains for this project (project setup via \`uv\`; Ollama

working; ingestion → chunking → embeddings → manual search → Chroma → retrieval → context →

generation, each inspectable; provider abstraction with one hosted provider pluggable; learning
modes (explain/quiz/Socratic) grounded in retrieved source material;

missing-info/retrieval-failure/generation-failure experiments done; chunk-size, overlap, and Top-K

experiments done; evaluation dataset + retrieval metrics + generation evaluation; observability;

tests; README with architecture and experiment write-ups; and the user can explain the entire

pipeline unaided). The project isn't "done" until the knowledge test in §9 has been given.

**## 11. Where this leads**

This pipeline is the foundation for later systems: RAG → Tool Calling → Agents → Agent Runtime →

Evaluation → Production AI Systems. Keep that framing in mind when explaining *\*why\** a given

architectural discipline (interfaces, observability, evaluation-first thinking) matters beyond

this one project.