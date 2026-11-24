# Step 21b — general documents

2026-11-24. The user's choices: a parser hand-built on PyMuPDF first (not pymupdf4llm or Docling),
public-domain test documents, .docx and .html with the standard library, GROBID optional and off
by default. **No new dependency.** Downloads approved by the user: 9 US federal publications
(public domain, 17 U.S.C. §105), 12.1 MB, fetched by `fetch_general_docs.py`; URLs, sizes and
sha256 in `general_docs_manifest.txt`. The files live in `data/raw/incoming/general/` (gitignored).

Three commits: A (the parser, `8bec336`), B (folders, Word, HTML, `86495c1`), C (this test set).

## A. The layout PDF parser (`src/rag/parse/pdf.py`)

`PDF_PARSER=pymupdf` (default) or `grobid`; `--parser` on `rag ingest` and `rag parse`. Ingest no
longer stops when GROBID is down. `rag parse <pdf>` prints what the parser decided: headings from
bookmarks or layout, body size, headers/footers and page numbers dropped, paragraphs joined across
pages, headings split off their paragraph, bookmarks it could not place.

What it does, in order: lines with size, boldness and position (rotated text skipped) → "needs OCR"
if there is almost no text → running headers/footers (margin text repeated on ≥3 pages and ≥10%)
and page numbers dropped → body size → headings that share a block with their paragraph split off
(the first lines, if bold or larger, or a bookmark title; also run-in bold headings like "**5.1
Setup** We now …") → a number on its own line ("4.") joined to the heading after it → title = the
largest text on page 1 if it is unique and ≥2 pt above the body → headings from the PDF's
bookmarks (matched on their words, numbering and "Chapter 3"/"Slide 2:" prefixes ignored, never a
footer that repeats a title), else from layout (larger, or bold throughout, or "2.1 Title"-style
numbering, or a short all-caps line) → paragraphs from blocks, hyphens and ligatures fixed, joined
across a page break → "Figure 2:" captions to their own section → References dropped, appendices
kept → one section per page if fewer than 2 headings.

**Cost of dropping GROBID, on the papers** (scratch copy, `parser_vs_grobid.py`, runs in
`parser_vs_grobid_runs/`):

| | GROBID | layout parser |
|---|---|---|
| hybrid Hit@1 / Hit@5 / MRR | 22 / 33 / 0.675 | 24 / 33 / 0.706 |
| fused Hit@1 / Hit@5 / MRR | 26 / 35 / 0.754 | 26 / 35 / 0.753 |
| gold quotes in one chunk | 40/40 | 40/40 |
| parse time per paper | ~1.6 s (a Java server) | 0.08–0.26 s |

The target was "within 2 questions of GROBID on Hit@5": it is equal. Biggest single move:
`adam_name` 1 → 14 (hybrid) because the layout parser keeps the Acknowledgements ("for coining the
name Adam", a fair near-distractor that GROBID's rules dropped) and the answering chunk is larger.
Known gap: text inside vector charts (axis labels) becomes a paragraph; GROBID drops it.

Gold quotes now match typography-insensitively (straight/curly quotes, dashes, Unicode minus,
ligatures, spacing, line-end hyphens): a quote names a passage, not its typesetting. No gold moved
on the existing sets.

## B. Folders, Word, HTML

- `rag ingest files <folder>` walks subfolders (symlinked folders not followed) and prints every
  file it skips and why: hidden or temporary (`.git`, `.DS_Store`, `~$draft.docx`), unsupported
  type, empty, over 200 MB, same content as another file. Files in a folder are keyed by their
  path there (`hr/leave.htm` → `file:hr-leave`), so equal names in different subfolders don't clash.
- Word (`rag/parse/office.py`, zipfile + ElementTree): headings from paragraph styles, list items
  as "- …", table rows as "cell | cell", title from document properties. Schema v5 adds 'docx'.
- HTML (html.parser): h1–h6 headings, page furniture (nav, header, footer, aside, script, style,
  form) dropped, a single `<h1>` is the title, table rows as "cell | cell".
- The file route downloads HTML and Word files instead of showing them (a stored page must never
  run on the app's origin) and sends `Content-Security-Policy: sandbox`.
- A local file's title falls back to its original name, not our stored copy's.

## C. The general-documents set (`eval/sets/gov-docs.jsonl`)

**Documents and the structure check by hand** (`rag parse` on each, all ingested in 3.3 s with
GROBID stopped):

| Document | Format, pages | Headings | What I saw |
|---|---|---|---|
| OPM handbook, childbirth/adoption/foster care (2025) | PDF, 146 | bookmarks 157/167 | Right tree (I: Introduction > Know Your Options > …). Some faux-bold text is drawn twice in the PDF and comes out doubled ("agencies agencies") |
| OPM handbook, elder care (2015) | PDF, 32 | bookmarks 54/59 | Right; a near-distractor for the one above (both give 12 weeks / 13 days of sick leave) |
| OPM fact sheet, safe leave | HTML | 12 sections | Right; one site menu heading ("Resources For") survives because the site doesn't mark it as `<nav>` |
| FEMA "Are You Ready?" (2020) | PDF, 36 | bookmarks 55/62 | Right; 32 running header/footer blocks dropped |
| OSHA 3151 Personal Protective Equipment | PDF, 46 | bookmarks 38/39 | Right; 76 header/footer blocks dropped |
| IRS Publication 463 (2025) | PDF, 62 | bookmarks 55/55 | Right after two fixes (headers that change per chapter; "4." on its own line). One "Actual Car Expenses" section is 9.6k words (depreciation tables); a running header occasionally lands inside a paragraph joined across pages |
| NIST SP 800-63B-4 (2025) | PDF, 129 | bookmarks 95/100 | Right after splitting headings off their paragraphs (10/100 before). Appendix B nests under §8 (its bookmark wasn't found) |
| NASA "AI for Lessons Learned" (2024) | PDF slides, 11 | bookmarks 7/11 | One section per slide title; screenshot text is noisy (body size 5.5 pt) |
| GSA/FICAM autopen decision template | DOCX | none | Correct: the template really holds 83 words; flagged suspect (79 body words) |

**Questions:** 37 (33 answerable, 4 unanswerable): direct, paraphrase (reworded, low word overlap),
identifier (numbers, standards, classes), a multi-step one, near-distractors across the two OPM
handbooks and the hard-hat classes, and traps for the unanswerables (a 2026 mileage rate when the
document is 2025; respirator fit-testing, which the OSHA booklet says it doesn't cover; paid
parental leave for active-duty service members). Drafted by Claude from exact sentences; **not yet
reviewed by the user**. After reading the retrieved chunks, the gold of two questions was widened
(tag `gold-widened`): the 146-page handbook states the 12-week and 12-month facts in two or three
places, and the chunks ranked above the gold answered the question.

**Retrieval** (all 33 answerable):

| | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| dense | 20 | 29 | 0.739 |
| BM25 | 13 | 28 | 0.591 |
| hybrid | 17 | 31 | 0.704 |
| **hybrid + fused rerank (the `ask` default)** | **21** | 30 | **0.773** |
| hybrid + replace | 20 | 31 | 0.748 |

Unlike the papers (where BM25 led), dense beats BM25 here: these questions are mostly reworded. The
fused reranker is again the best default. Genuine misses: `osha_hazard_assessment` (heavy rewording:
"dangers", "protective gear"; rank 13–20) and `elder_combined_limit` (the 13-day chunk outranks the
12-week cap; rank 8 with fused).

**Answers** (`rag eval answers --set gov-docs`, fused, 1 run, 6.7 min): automatic checks — key facts
29/33, gold given to the model 32/33, refusals right 4/4, refusals citing sources 0, one guardrail
flag. Read by hand (11 answers): of the 4 marked "wrong?", 3 are correct (the key-fact check misses
rewordings, as in step 16) and 1 is **partial** (`irs_cruise`: the model gave the $2,000 rule for
conventions on cruise ships, a neighbouring rule; the general daily limit wasn't used). The flag
(`fema_flood_wait`, "30-day" not in the cited source) is a true catch: the answer is right but cites
the wrong source. The active-duty trap was refused. Median 11.2 s per answer, 7.7 s of it Ollama
reading the 3,200-token prompt.

**The gate** now runs the gov-docs set and replays its answers. Adding 1,004 chunks changed the
corpus-wide BM25 statistics: two injection questions moved from rank 1 to 2 inside their own
two-chunk documents, and papers Hit@1 rose 22 → 24. Baseline updated with that reason.

## Open questions for the user

- **BM25 statistics per collection or shared?** Today a new folder changes the ranking in every other
  collection (seen above). Per-collection statistics keep collections independent; shared ones are
  simpler and what "search everything" needs. A design decision.
- **PyMuPDF is AGPL** (noted in `rag/parse/pymupdf.py`). Fine for an open-source repo; a closed
  packaged release needs a commercial licence or another PDF library. Decide before step 28.
- **The prompt still says "paper" once** (v3 rule 7: "the paper, model or subject"). Left alone on
  purpose: changing a measured, versioned prompt needs a v4 and its own answer eval.
- **The gov-docs questions were written by Claude**; a review of a sample by the user is pending.

## Limits

- No OCR (a scanned PDF is reported as "needs OCR"), no tables as structure, no PowerPoint/Excel,
  English only.
- Slide decks parse, but their text is often screenshots; answers from them are weaker.
- The real `papers` collection is still parsed by GROBID (re-parsing it with the layout parser was
  only done in a scratch copy).
