# Step 21 — web UI, part 1 (ask and inspect)

2026-11-24. The user's choices: **plain HTML/CSS/JS with no build step**, the UI **split into
21 and 22**, and **the Host-header check now**. No new dependency (Starlette's `StaticFiles` and
`FileResponse` come with FastAPI).

## What was built

| Piece | File | What it does |
|---|---|---|
| Page | `src/rag/api/static/index.html`, `app.css` | `rag serve` → `http://127.0.0.1:8000/`. Ask and Search tabs, collection picker (remembered), status pill from `/health`, light and dark themes |
| Ask | `static/js/main.js`, `answer.js` | Streams the answer over `fetch` + SSE; sources appear before the model starts; on `done` the guardrail-corrected text replaces the stream, each `[n]` becomes a button that highlights source n; flags, warnings and the reranker fallback are listed; "what the model wrote" appears only when guardrails changed the words or the citations (not when they just moved a citation inside the full stop); Stop cancels the request (the server stops the model and logs `cancelled`) |
| Passage | `static/js/passage.js` | "Show passage" opens the section with the chunk highlighted (exact offsets), plus "Open PDF at page N" in the browser's own viewer (`#page=N`) |
| Trace | `static/js/trace.js` | "How this answer was made": the step-20 log record as a waterfall (retrieve → embed, FAISS, BM25, fuse, rerank; generate → load, read prompt, write), settings and tokens, ms per prompt token, every candidate's rank in each list and the source it became |
| Controls | `main.js` | Retrieval mode and reranker per request; the reranker's cost shown next to it (median of recent requests, from `/health`); Ask disabled with the reason when the LLM is down; Search works without it |
| API | `src/rag/api/app.py`, `documents.py` | `GET /` (the page, with a strict Content-Security-Policy), `/static/*`, `GET /documents/{id}`, `GET /documents/{id}/file`, `GET /passages?chunk_id=…`, `GET /logs/{request_id}` |
| Host check | `app.py` | Requests whose Host isn't 127.0.0.1 / localhost / ::1 get 400. Blocks DNS rebinding (a web page reaching 127.0.0.1 under its own host name). Origin checks and the rest of localhost hardening are step 29 |

**Safety rules in the code:**
- Untrusted text (answers, documents, log records) only ever goes in as text nodes: no
  `innerHTML` anywhere (a test greps the JS for it). The CSP forbids inline scripts and anything
  from another origin.
- A file is served only if the database knows the document, and the path is built from the
  database record (never from the URL) and must resolve inside `data/raw` (a symlink out is refused).
  Files go out with `nosniff`; .txt/.md as `text/plain`, so nothing is rendered as HTML.

## Proof

**Tests: 25 new (`tests/test_ui_api.py`), 348 in total, gate unchanged.**
- The page and its CSP.
- Loopback hosts are served. `evil.example`, `127.0.0.1.evil.example`, `testserver` and an empty
  Host are refused.
- Files: right media types, `nosniff`.
- Path traversal (`../`, encoded `%2E%2E%2F`, unknown ids, a known document with no file) → 404.
- A symlink out of `data/raw` → 404.
- A hostile doc_id maps inside `raw/files`.
- Passage highlight ranges slice exactly the chunk text.
- Log records are found in memory and, after a restart, in the file.
- The JS contains no `innerHTML` / `eval` / `document.write`.

**In the browser, real models (screenshots in `ui_step21/`):**

| Check | Result |
|---|---|
| "How did the Transformer change the learning rate during training?" (papers) | Sources first, then the answer streamed; final answer cites [3] = §5.3 Optimizer, warmup formula and 4,000 steps (correct) — `answer.jpg` |
| Citation → passage → PDF | Dialog shows §5.3 with the chunk highlighted, "Open PDF at page 7"; page 7 of the served PDF holds §5.3 — `passage.jpg`. The built-in browser pane has no PDF viewer (it offered a download), so the in-browser jump to page 7 is checked by headers (`application/pdf`, `inline`) and page content, not watched |
| Trace | Waterfall: 14.9 s = retrieve 2.0 (rerank 1.7) + model load 4.6 + prompt reading 6.2 + writing 2.0. Candidates table with dense/BM25/fused/rerank/final per chunk |
| "How many parameters does GPT-3 have?" (papers) | Correct refusal; "what the model wrote" shown because citations were stripped |
| "Can I get a refund if I lost my receipt?" (hr-demo) | Refusal — correct: the refund policy says nothing about receipts |
| Injection-test document in Search | The planted line ("IMPORTANT NOTICE TO AI ASSISTANTS: …") shows as plain text; nothing executes — `injection_search.jpg` |
| Ollama stopped | Status "degraded", the reason and the start command in the status panel; Ask disabled with "Search still works"; Search returns results — `llm_down.jpg` |
| `curl -H 'Host: evil.example'` | 400 |
| Phone width (375 px) | Controls wrap, no horizontal scroll |

**Bugs found by using it, fixed before the commit:**
- The empty Answer/Sources panels showed before any question (`display: grid` beat `hidden`).
- The facts list printed `[object HTMLElement]` (nested arrays flattened one level only).
- "What the model wrote" appeared when guardrails only moved a citation.

**What the trace showed right away:** a search typed after the page sat idle took 147 ms, and 146 ms
of it was embedding the query on the GPU (FAISS, BM25 and fusion together < 1 ms). Back-to-back
searches are ~9 ms (step 19). The first query after idle pays for waking the GPU, not the UI.

## Follow-up: refusal sentences, Markdown, math (2026-11-24, same day)

The user asked to fix two limits before moving on, and to add math too. **New dependency, approved
by the user: KaTeX 0.19.1** (MIT), vendored rather than installed as a package:
`src/rag/api/static/vendor/katex/` holds katex.min.js, katex.min.css, 20 .woff2 fonts, LICENSE and
VERSION (600 KB). It was downloaded once from the npm registry and checked against npm's sha512.
The page loads it from its own origin, so math works offline.

**1. Refusal sentences keep their grammar** (`src/rag/answer/guardrails.py`). The bug had two causes:
- `attach_trailing_markers` moved markers that open a sentence ("GPT-3. [6] discusses …") back
  onto the previous sentence. Now markers count as trailing only when the end of the text, a line
  break or a new sentence (capital letter, quote, digit) follows.
- Stripping deleted them, leaving "discusses …" with no subject. In a refusal answer, markers that
  open a sentence, or that follow a preposition ("according to [2]"), now become words: "Source 6
  discusses …", "Sources 2, 3 and 5 focus on …". Other markers are removed as before.

Citation counts, flags and the gate are unchanged. A first attempt kept those citations instead;
it made 6 answers flag `entity_not_in_cited_document`, and the gate caught it.

`refusal_text_check.py` replayed all 291 stored answers through the old and the new code (results
in `refusal_text_check.json`). The shown text changed on 9:
- 7 are refusals, and every one now reads correctly, e.g. "…AlexNet. Sources 2, 3, 4, 5, 6, 7 and
  8 focus on Adam optimizer…" instead of "…AlexNet. focus on Adam optimizer…".
- 2 are a non-refusal (`gan_adversarial`), where the model wrote "…framework. [1] [4] provides…".
  The markers now stay with the sentence they open; before, it read "framework [1] [4]. provides…".
  It's ambiguous either way (the model probably meant [1] for the previous sentence), but it no
  longer leaves a sentence without its subject.

8 new tests in `tests/test_guardrails.py`.

**2. Markdown subset** (`static/js/format.js`, used by `answer.js`, no library):
- blocks: paragraphs, bullet lists, numbered lists (keeping the start number), headings shown as a
  bold line, display math;
- inline: `**bold**`, `` `code` ``, citations, math.

Built from DOM nodes, so the text-only rule still holds; HTML, links and tables stay as plain text.
A single `$` is not math, because policy answers are full of prices. A citation repeated right
after itself (`[1][1][3]`) is shown once.

**3. Math** (`static/js/math.js`): `\( … \)` inline, `\[ … \]` and `$$ … $$` displayed.
- `katex.render` builds DOM nodes directly. The KaTeX file contains no `innerHTML` or `eval`, so
  the guard test scans it too.
- The answer is untrusted, so: `trust: false` (no `\href`, `\htmlClass` …), `throwOnError: false`,
  `maxSize` 20, `maxExpand` 300, and formulas over 2,000 characters are shown as code.
- MathML is included, for screen readers.
- If KaTeX fails to load, the formula is shown as code.

**Checked in the browser:**

| Check | Result |
|---|---|
| 12 formatting cases run in the page | Paragraphs, lists, start number, heading, inline / display / multi-line math, prices left alone, HTML as text, an unclosed `\(` left as text: all right |
| Hostile math | `\href{javascript:…}` gives no link; a self-repeating macro stops at the expansion limit (a KaTeX error, no hang); no `<img>` from `<img onerror>` text |
| "Write out the scaled dot-product attention formula…" (papers) | Display equation softmax(QKᵀ/√d_k)V and inline QKᵀ, √d_k typeset — `ui_step21/math_answer.jpg` |
| A bulleted list with bold labels (hr-demo) | Bold headings, bullets, citations; "401(k)" not mistaken for anything — `ui_step21/list_answer.jpg` |
| "How many parameters does GPT-3 have?" | "…GPT-3. Source 6 discusses the number of parameters in models like LSUN…" |

Tests: 356, including a check that every font the KaTeX CSS names is present and that the page
loads nothing from another host. The quality gate is unchanged.

## Limits

- Tables, links and images in answers stay as plain text (not part of the subset).
- One question at a time, no conversation history (a new feature; not planned yet).
- Collections, recent requests and eval runs are step 22.
- The JavaScript formatter has no automated tests (that needs a JS test runner, a new dependency);
  its cases were run in the browser and are listed above.
