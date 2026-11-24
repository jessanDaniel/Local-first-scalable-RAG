# Step 22 — web UI, part 2 (collections, activity, evaluation)

2026-11-24. The user's choices (all as recommended in the pitch):
- **Evaluation in the UI is view-only.** Runs are still made with the CLI.
- **The collections overview is the Documents tab's landing view.**
- **Answer runs get a read-only summary.**
- **The Activity window defaults to the last 1,000 records, from all sources.**

No new dependency.

## What was built

| Piece | File | What it does |
|---|---|---|
| Collections overview | `static/js/overview.js`, `GET /collections` | The Documents tab with no collection picked: one row per collection. Shows documents, chunks, parse doubts (suspect / failed), folders (missing ones in red), last sync, and the newest retrieval run of each eval set that covers it. A name opens the collection's management view; "‹ All collections" comes back. `/collections` gained `n_suspect`, `n_failed`, `n_folders`, `n_missing_folders`, `last_synced`, `created_at` |
| Activity: stats | `static/js/activity.js`, `GET /logs/stats` | `rag logs stats` as tables, per route. Counts, error / cancel / fallback / refusal / flag numbers, and p50 / p95 / max per stage, with the sub-stages indented and a bar (p50 dark, p95 light, against the p95 total). Shows the Ollama prompt-reading ms/token. Window: last 100, last 1,000, or every record (`last=0`). Source: app, CLI or both. The time span of the window is shown above the tables |
| Activity: requests | `activity.js`, `GET /logs` | Newest first. Filters: route, status, source, "flagged answers only". 50 per page with "Show older". A row opens the step-21 trace view, plus the question, the answer and its flags |
| Evaluation: runs | `static/js/evaluation.js`, `GET /eval/runs` | Retrieval runs of one set: Hit@1/5/10, MRR, p50 latency, chunks, commit. Retriever and rerank are normalised for old runs: before step 13 = dense; a retriever of `rerank` before 15b = hybrid + replace |
| Evaluation: one run | `GET /eval/runs/{id}` | Config, results by question type, and every question's first gold rank, coloured: 1, 2–5, 6–20, miss. Each row has the question's wording from the current set file, word overlap, and the top-1 document. A "gold outside the top 5" filter. Unanswerable questions show what came first anyway |
| Evaluation: A → B | `GET /eval/compare?a=&b=` | `rag eval compare` as data: config changes, metric deltas (coloured; slower latency is red), and the questions whose rank moved, with the top-1/5/10 lines they crossed and their wording |
| Evaluation: answer runs | `GET /eval/answers` | Model, prompt, guardrails, sanitising, retrieval, number of answers. The automatic checks: key facts, gold cited, unanswerable refused, refusals that cite, flags, injections followed, median seconds. Labelled "automatic checks only" |
| Evaluation: gate | `GET /eval/gate` | `eval/baseline.json`: reason, commit, config, Hit@k / MRR per set, the questions allowed to stay outside the top 5, how many replayed answers carry a flag, and the update history (newest first) |
| Shared code | `src/rag/eval/runner.py`, `src/rag/eval/browse.py`, `src/rag/requestlog.py` | `compare_runs` became `run_diff` (data) + a text renderer, so the CLI and the API can't drift. `run_brief` / `retriever_and_rerank` for run lists. `browse.py` holds the read-only readers: run ids are looked up in the directory listing and never joined into a path; parsed files are cached by (mtime, size). `requestlog` gained `iter_newest` (newest first, line by line), `matches` / `newest` (filters, paging), `brief` (a list row) and `for_stats` (only what `stats` reads) |
| CLI | `rag logs stats --source api\|cli` | The same source filter as the UI, for parity |

## Proof

**Parity with the CLI on the real data** (scratch scripts, outputs compared field by field):

| Check | Result |
|---|---|
| `rag eval compare` text, before vs after the `run_diff` refactor, 4 real pairs (hybrid → fused, gov-docs off → fused, hybrid-timed → grobid-off, dense baseline → rerank-bge) | byte-identical ×4 |
| `/eval/compare` JSON vs `run_diff` of the same two runs loaded by the CLI, same 4 pairs | identical ×4 |
| `/logs/stats?last=1000` vs `rag logs stats --last 1000 --json`, captured before any change | identical |
| `/logs/stats?last=0` vs `rag logs stats --last 100000 --json` | identical |
| `/logs/stats?last=200&route=ask` vs `--route ask` | identical |

**Tests: 24 new in `tests/test_activity_eval_api.py`, 437 in total, gate unchanged.**
- Requests: newest first, with filters and pages; with file logging off the list comes from memory; with `LOG_TEXT=false` no question is shown.
- Log reading: rotated files are read in reverse order, and a line cut short is skipped.
- Stats: API = CLI for every record, the last 3, and route=ask; the slimmed records give the same stats as full ones.
- Runs: newest first and filtered by set; old runs are normalised; run detail; compare = `run_diff` and the CLI text.
- Compare errors: different sets → 422, an unknown id → 404, a missing parameter → 422.
- Run ids `..`, `...json`, `..%2Fbaseline`, `%2E%2E`, `%2Fetc%2Fpasswd`, `papers:latest` and `../secret` (a real file one directory up) → 404.
- A broken run file is skipped, not an error.
- Answer runs: the summary is listed, with no per-answer text.
- Gate: 404 before a baseline; after two updates, flags are kept from the first and history is newest first.
- The collection overview's fields; the page has the new tabs; static files are revalidated.

**Size and cost:** a synthetic full-size log (6 files × 20 MB = 119 MB, 65,603 real records with new ids, 1 ask per 20 searches). Measured in-process (`log_timing.py` in the scratchpad):

| Call | Time |
|---|---|
| `/logs` newest 50, no filter | 8 ms |
| `/logs` newest 50 asks | 13 ms |
| `/logs` with a filter nothing matches (scans all 119 MB) | 431 ms |
| `/logs/stats` last 1,000 | 18 ms |
| `/logs/stats` every record, first version | 1,088 ms, **+650 MB** peak memory |
| `/logs/stats` every record, with `for_stats` | 621 ms, +105 MB |

The first version held every full record (candidates, answer text) while computing the stats. The server already uses 4.2 GB and the Mac swaps, so `/logs/stats` now keeps only the fields `stats` reads. The numbers are identical (tested).

On the real data, warm: run list 0.4 ms (15 ms cold, 38 files), the 4 parity pairs 15 ms each, `/eval/gate` ~100 ms cold.

**Browser check** (real data, Ollama off), screenshots in `ui_step22/`:
- Activity: stats for every record (`activity_stats_all.jpg`) show ask prompt reading at a median 2.34 ms/token, as in step 20. A request row opens its trace (`activity_request_trace.jpg`).
- Evaluation: the papers hybrid → fused comparison shows Hit@1 +4, MRR +0.080 and 14 questions moved (`eval_compare_hybrid_vs_fused.jpg`), as `rag eval compare` does. Also `eval_run_detail.jpg` (by type, ranks) and `eval_answers_and_gate.jpg`.
- Documents: the overview (`docs_collections_overview.jpg`) shows gov-docs 5 suspect, injection-test 6 suspect, one folder synced. Opening a collection and "‹ All collections" both work.
- Phone width (375 px): every tab fits; tables scroll inside their box (`activity_phone_375.jpg`).
- No console errors.

## Bugs found by the browser check (fixed)

1. **"All collections" fell back to the remembered collection.** `loadCollections("")` treated an empty pick as "no pick" and re-selected the stored name. Now an explicit "" means All, and it is remembered.
2. **Overlapping loads drew twice.** Opening Activity and changing the window at once showed three route blocks (two windows). Every panel that clears, awaits, then appends now ignores a response a newer request has replaced: stats, the request list, run detail, compare, answer runs, the gate and the overview.
3. **The browser ran stale scripts.** After the fix, the page still raced. Static files had no `Cache-Control`, so the browser reused the old `activity.js` from its cache without asking. `/static/*` now sends `Cache-Control: no-cache`: the browser revalidates by ETag (a 304 when unchanged), so an updated app never runs yesterday's scripts against today's page. Tested.
4. **Five tabs widened the page on a phone** (467 px layout on a 375 px screen). The tab bar now scrolls sideways.

## What the views show that was hard to see before

- **The default stats window is dominated by the step-24 load measurement.** The last 1,000 records span 13 seconds, all `/search` from the sync test. That is why the span is printed above the tables. "Every logged request" gives the real mix (14 asks, 1,397 searches).
- **"Latest retrieval eval" on the overview is the newest run, not the default config.** For gov-docs that is a `hybrid + replace` experiment. The config is printed next to the number for that reason. The gate panel has the committed reference numbers.
- **The gate's allowed misses are visible.** papers keeps 7 questions outside the top 5 (attn_heads, vae_sgvb, bn_gamma_beta, resnet_degradation, s2s_bleu, w2v_cbow, resnet_ensemble_error); gov-docs keeps 3. These are the step-23/26 targets.

## Limits

- **Pages use offsets.** New requests arriving between "Show older" clicks shift rows, so a row can repeat. Fine for one user on one Mac; a `before=<request id>` cursor would fix it.
- **A filter that matches nothing scans the whole log** (0.43 s at the 120 MB limit). There is no index over the log; JSON lines were chosen on purpose in step 20.
- **Question wording comes from the set file as it is now.** A run made before a set was edited shows today's wording (the panel says so). The ids and ranks are the run's own.
- **No per-answer view of answer runs** (labels stay in `eval/labels/` and `rag eval sheet`).
- **The JavaScript has no automated tests** (that needs a JS test runner, a new dependency). Its behaviour was checked in the browser, as listed above.
