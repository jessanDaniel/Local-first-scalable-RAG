# Step 11 — collections, a second corpus, the title fix

2026-11-20. Three commits: 11a collections, 11b text/Markdown + `hr-demo`, 11c titles.

## 11a. Collections

- `collections` table. Every document belongs to exactly one collection; ids are slugs.
- Schema v4. The v3 → v4 migration rebuilds the `documents` table: SQLite cannot add a
  foreign-key column or change a CHECK in place. It runs in one transaction with foreign keys
  off (otherwise dropping the old table cascade-deletes every section and chunk), and checks
  `PRAGMA foreign_key_check` before committing. Existing documents go to `papers`.
- `open_db` copies the database to `data/archive/rag-v<old>-<UTC>.db` before migrating.
  `rag status` opens the database read-only and reports a pending migration instead of running it.
- `rag ingest arxiv|files --collection NAME` (required; created if new). A document already in
  another collection is refused, never moved as a side effect.
- `rag search|ask --collection NAME` (repeatable). The filter runs inside FAISS
  (`IDSelectorBatch`), so a small collection still gets its full top-k.
- `rag collections`; `status` shows documents per collection.

**Real database:** migrated on a copy first, then for real. Same 10 documents, same 400 chunk
ids (hash compared), and the existing index was still valid, so nothing was rebuilt.

## 11b. Text and Markdown, and the `hr-demo` corpus

- `TextParser` (`rag/parse/text.py`). Markdown: `#` headings give the tree, and a single opening
  `# Title` is the title. Plain text: a heading is one line of at most 12 words and 80
  characters, starting with a capital or a digit, with no closing punctuation, and followed by
  more text. When unsure, it's a paragraph.
- Formatting rules:
  - Wrapped prose is joined into one line.
  - Lists, code and blocks of short lines keep their line breaks.
  - A line under the title stays a paragraph even when it looks like a heading. A test found
    this bug: "Version: 2" became a section.
- `rag ingest files` takes `.pdf`, `.txt`, `.md`. Text never needs GROBID.

| Document | Sections | Chunks | Title |
|---|---|---|---|
| company_policy.txt | 8 (Section 1–8) | 8 | Acme Corp Customer Policy Handbook |
| employee_handbook.txt | 21 (Preamble, Section 1–18, Appendix A–B) | 21 | Acme Corp Employee Handbook |

The index rebuild embedded only these 29 chunks; the 400 paper vectors came from the cache.

**Phase 15 questions vs the chunks** (`hr_quotes_vs_chunks.py`, gold ids in
`hr_quotes_vs_chunks_gold.jsonl` for eval v0):

| | Quotes | In exactly one chunk | Split across chunks | Missing |
|---|---|---|---|---|
| answer quotes | 31 | 31 | 0 | 0 |
| distractor quotes | 10 | 10 | 0 | 0 |

Every section is shorter than one chunk (≤169 tokens), so chunk = section and nothing can split.
That means this corpus cannot reproduce the Phase 14 boundary failure. Testing that needs
longer documents.

**Collection filter:** top-1 is a gold chunk for 22/24 answerable questions **with and without**
`--collection hr-demo`. No paper chunk ever ranked first for an HR question. The two domains are
too far apart to compete, so this pair of corpora shows that the filter works, not that it helps
retrieval. Its value here is scoping, and access control in Stage 5.

## 11c. Titles

`rag/parse/title.py`:
- `title_problem()` rejects:
  - licence and copyright notices (notice phrasing, not single words)
  - venue and review banners
  - arXiv stamps
  - proceedings headers
  - emails and URLs
  - authoring-tool placeholders
  - text that is empty, too long or mostly not letters
- For local files the candidates are tried in order: parser → PDF metadata → largest horizontal
  text on page 1 (rotated stamps skipped, small-caps fragments joined, ligatures expanded) →
  file name.
- A guessed title marks the document `suspect` ("title guessed from …") and keeps the rejected
  candidates in `metadata.title_rejected`.

Checked on all 10 papers, using the titles GROBID gave earlier:

| Paper | Chosen from | Title |
|---|---|---|
| 8 papers | GROBID | correct |
| Attention (1706.03762) | largest font | Attention Is All You Need (GROBID: licence line) |
| Adam (1412.6980) | PDF metadata | Adam: A Method for Stochastic Optimization (GROBID: ICLR banner) |

10/10 correct. 9/10 match the arXiv API title exactly. The GAN PDF itself says "Generative
Adversarial Nets", while arXiv lists "…Networks".

Live run with GROBID on the two step-9 PDFs (`rag ingest files … --collection papers` in a
scratch folder): "Attention Is All You Need" (largest font) and "Adam: A Method for Stochastic
Optimization" (PDF metadata), both `suspect` with the reason recorded.

**Gap found and fixed along the way.** The index check compared only chunk ids, but every chunk
is embedded with a "title | section path" header, and the id covers neither. A corrected title
would therefore have left stale vectors with `status` saying "current".

- Every build now records a fingerprint of all (chunk id, title, section path).
- Loading and `status` compare it against the database.
- A test changes a title and expects the index to be stale.
- The real index was rebuilt to record the fingerprint: 0 chunks re-embedded, all 429 from the
  cache.

## Numbers

- Tests: 139 → 194.
- `scripts/check.sh` passes.
- Real database: schema v4. Collections `papers` (10 documents, 400 chunks) and `hr-demo`
  (2 documents, 29 chunks). The index is current.
