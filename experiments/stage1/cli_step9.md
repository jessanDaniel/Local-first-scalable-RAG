# Stage 1 step 9 — one `rag` command

2026-11-20.

## What was built

- `src/rag/main.py`: the `rag` command. Every argument is defined here; each command's
  code (`run(args)`) is imported only when that command runs.

  ```bash
  rag status [--json]
  rag ingest arxiv|files|tei ...
  rag parse <pdf>
  rag chunk [--show DOC_ID]
  rag index build|list
  rag search "<q>" [-k] [--explain]
  rag ask "<q>" [--explain]
  ```

  `--data-dir DIR` points everything at another data folder. The old `python -m rag.<module>`
  entry points still work and go through the same parser (`run_module`).
- `src/rag/status.py`: what is stored (documents by status and source, sections, chunks, chunker
  version vs settings, parsed-but-unchunked documents), the CURRENT dense index and whether it
  is stale, whether GROBID and the LLM answer (1 s probes), and the next command to run. Never
  writes: a missing database is reported, not created. This report is the basis for `/health`
  in Stage 3.
- `src/rag/index/current.py`: the CURRENT-build check, using only the JSON files (no FAISS).
  `dense.load_current` uses the same check.
- `db.stats()`. Tests: `tests/test_cli.py` (23 new, 139 in total).

## Numbers

| Check | Result |
|---|---|
| `rag --help` | 0.69 s wall (including `uv run`) |
| `rag status` | 0.32 s wall, including both probes |
| torch / FAISS / NumPy / sentence-transformers loaded by `--help` or `status` | none (tested) |

## End-to-end run in a scratch data folder

`rag --data-dir <scratch>`, two local PDFs (Attention, Adam), with `status` after each step:

| Step | Result | `status` "next" afterwards |
|---|---|---|
| (empty folder) | no database, nothing created | `rag ingest arxiv …` |
| `rag ingest files` | 2 ok, 0 fallback, 9.9 s + 1.7 s parse (first is GROBID warm-up) | `rag chunk` |
| `rag chunk` | 84 chunks, median 200 tokens, max 350 | `rag index build` |
| `rag index build` | 84 embedded on MPS in 1.4 s, 126 KB | `rag ask "<question>"` |
| `rag search` (Adam bias correction) | §3 Initialization bias correction at ranks 2–3 | |
| `rag ask` (√d_k) | correct, cites [4] = §3.2.1, p.4; 25 s including model load | |

## Found along the way

1. **Local-file titles are wrong.** For PDFs ingested with `rag ingest files`, GROBID read the
   arXiv licence line ("Provided proper attribution is provided, Google …") and the venue banner
   ("Published as a conference paper at ICLR 2015 ADAM: …") as titles. arXiv ingests are
   unaffected because their titles come from the arXiv API. The title also goes into every
   chunk's embedding header and every citation label, so this matters for the generic pipeline.
   Fix with the second-corpus work (§0.1 item 3): title quality gates and a cleanup rule, tested
   on these two PDFs.
2. **Proxies and localhost.** In this sandbox, `HTTP_PROXY` is set and `localhost` is not in
   `no_proxy`. The proxy answered GROBID's health check with 503, so a stopped GROBID looked
   "unhealthy". The status probes now skip environment proxies for localhost / 127.0.0.1 / ::1.
   The ingest client did not hit this in practice (the proxy forwards when GROBID is up), but it
   is worth the same treatment if it ever does.
