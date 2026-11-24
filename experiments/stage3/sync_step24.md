# Steps 24 + 30 — add documents from the UI, kept in sync, no restart

2026-11-24. The user's choices:
- BM25 statistics per searched collection;
- a Rescan button plus a rescan at server start (no file watching);
- uploads as raw file bodies (no `python-multipart`);
- an Origin check now.

**No new dependency.** Three commits: A `478ca2b`, B `dbcb234`, C (this one).

## A. BM25 statistics over the searched collections (`src/rag/index/bm25.py`)

With a collection filter, N, the average length and every df are counted over the searched chunks
only, at query time (the postings are already filtered there). Without a filter, the statistics
are the corpus's, exactly as before. This is simpler than the pitched "an index per collection",
with the same effect. Rebuilding all of BM25 from SQLite takes milliseconds at this size (the swap
below, 2,447 chunks, took 0.33 s including FAISS).

Gate: papers and injection return **exactly** to their numbers before gov-docs was added (papers
Hit@1 22, MRR 0.6753; injection 9/9 at rank 1). Cost: one hr-demo question moved 1 → 2. It is a fused
near-tie (0.0325 vs 0.0323) in a 29-chunk collection, where statistics are noisier.

## B. Sync (`src/rag/sync.py`, `rag sync COLLECTION [--add-folder PATH] [--remove-folder PATH]`)

Folders are registered per collection (schema v6, `source_folders`); uploads live in
`data/raw/uploads/<collection>/`. A sync does the following:
- walks the folders recursively;
- skips files whose content hash is unchanged;
- parses new and changed files;
- removes documents this sync didn't produce (their file is gone, now skipped, or now maps to
  another id);
- chunks only the re-parsed documents;
- makes a new dense build from the embedding cache, only when something changed;
- prunes builds beyond the newest 3.

Safety rules:
- a folder that is missing right now keeps its documents (an unplugged disk);
- documents added any other way are never removed by a sync;
- a file that fails keeps its old document;
- a cancelled sync starts no new file, removes nothing, and still chunks and indexes what it
  finished, so the database and the index agree.

Document ids are global, so a file whose plain id another collection already uses gets the
collection name in front (`file:office-demo-safety-osha-…`).

`Pipeline.reload()` swaps in the new indexes on the pipeline thread while every model stays loaded.
Search leaves out chunks deleted mid-sync instead of failing.

## C. API, jobs, UI

- **Jobs** (`src/rag/api/jobs.py`): one sync at a time on a worker thread with its own database
  connection. The embedding model and tokenizer load lazily, so a rescan of unchanged folders takes
  0.36 s instead of ~10 s. Progress is a list of events; there is a cancel; the pipeline is swapped
  before a job is marked done. At server start, every collection with folders is rescanned.
- **Routes:**
  - `POST /collections`, `GET /collections/{name}` (folders, documents and their origin, running
    jobs);
  - `POST|DELETE /collections/{name}/folders`, `POST /collections/{name}/sync`;
  - `PUT /collections/{name}/files/{filename}` (raw body, streamed to disk, 200 MB limit, name
    rules, no partial files left);
  - `DELETE /documents/{id}` (uploads and CLI-added documents; a folder's document says to delete
    the file there instead);
  - `GET /jobs`, `GET /jobs/{id}?events_from=N`, `POST /jobs/{id}/cancel`.
- Management routes use their own short-lived connections, so they work before any index exists (a
  fresh install); the first sync then loads the pipeline. A folder inside the app's data folder (or
  containing it) is refused.
- **Origin check:** a POST/PUT/DELETE with an Origin that isn't this app gets 403. Without it, any
  web page could make the app index a folder of its choosing with a plain form post; the Host check
  doesn't catch that. Requests without an Origin (curl, the CLI) are let through.
- **UI, Documents tab** (`static/js/docs.js`):
  - create a collection;
  - the collection's folders with Add (a typed path), Rescan, Remove;
  - an upload drop zone and file / folder pickers (unsupported files skipped in the page);
  - live job progress with Cancel;
  - the document list (from folder / upload, chunks, status) with Remove for uploads.

## Proof

**Tests: 409** (25 new):
- sync: first sync, nothing changed, a changed file, a deleted file, a missing folder, documents
  added another way, uploads, cancel, remove-folder, prune, the id prefix, a failing file;
- reload: a running pipeline answering from new documents, chunks deleted before the swap;
- API: the Origin rules, collection, folder and upload rules, upload → sync → delete, a folder
  document refusing deletion, jobs, the startup rescan.

**In the browser, real models** (a demo folder: OSHA PDF, OPM HTML page, a Markdown policy):

| Check | Result |
|---|---|
| Create collection "office-demo", add the folder by path | Searchable after **10.7 s** (parse 0.2, chunk 1.9, index 8.4, mostly loading the job's embedding model the first time). Only 2 of 98 chunks needed embedding: the cache recognised passages identical to gov-docs. The OSHA file got the prefixed id |
| Ask "Can a visitor walk around the office on their own once they have signed in?" | "No, a visitor must be escorted beyond the lobby…" citing [1][2], with no restart |
| Edit the policy (floors 1–2 unescorted, floor 3 escorted), Rescan | **0.33 s**: 1 updated, 1 chunk embedded, swap 0.21 s. The answer now uses the new rule but over-generalises ("Yes…") and leaves out the floor-3 escort: a generation weakness, not the sync |
| Drop `fire-safety.md` and `budget.xlsx` | The .md uploaded and synced (0.36 s), listed as "upload"; the .xlsx skipped in the page |
| Delete the file in the folder, Rescan; Remove the upload in the UI | Removed (0.16 s and 0.18 s); gone from the list and from search |
| Search while a 9-file sync runs (`sync_latency.py`) | **0 errors**; p50 8.2 → 13.7 ms, p95 9.3 → 17.1 ms, one stall of 1.75 s. Throughput dropped while parsing (156 searches in ~6 s vs ~120/s idle): PDF parsing holds Python's interpreter lock (GIL), so the two threads take turns |

The demo collection was removed afterwards. The gate is unchanged.

## Found along the way, and fixed the same day

- **A false refusal (fixed).** "The question does not specify a particular floor" counted as a
  refusal sentence (a sources word plus a negation), so its citation was stripped, leaving "Source
  mentions". Negations whose subject is the question, query, request, user or "you" no longer count
  (`NOT_ABOUT_SOURCES` in `rag/answer/guardrails.py`). A marker right after the word "Source(s)"
  in a real refusal now keeps its number ("Source 6 discusses …"). Replayed on all 328 stored
  answers: 0 changed refusal status, 0 changed shown text, no flag changed (the gate passes). The fix
  is narrow: it only affects sentences like the one found.
- **No way to delete a collection (fixed).** `DELETE /collections/{name}`, `rag collections delete
  NAME [--yes]` (without `--yes` it only says what would go), and a "Delete this collection" button
  with a confirmation. It removes the collection's documents, chunks, the app's stored copies, its
  uploads and its folder registrations; files in its folders are never touched. The index is rebuilt
  by a new job type (`rebuild`) and swapped in. It is refused with 409 while a job for that collection
  runs. Checked in the browser: the collection leaves the picker; rebuild 0 new embeddings, swap
  0.18 s.

Still open:
- **Folder uploads flatten names:** a browser folder upload gives file names without their
  subfolder, so two files with the same name in different subfolders overwrite each other in the
  upload folder. Typing the folder's path (synced in place) has no such limit.
- **The embedding cache keeps vectors for chunks that no longer exist**; it only grows. Fine at this
  size, worth pruning later.

## Limits

- No automatic file watching (Rescan, or the rescan at server start).
- One job at a time; a big folder (hundreds of PDFs) takes minutes, with progress and cancel.
- Search slows (but keeps working) while a sync parses (the GIL); measured in step 27's load test.
