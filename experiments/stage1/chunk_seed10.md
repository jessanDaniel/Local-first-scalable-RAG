# Stage 1 step 6 — chunker v2 on the first 10 papers

2026-11-18. Code: `src/rag/chunk/chunker.py` (`CHUNKER_VERSION = "v2"`, runs as `v2-350`).
Command: `uv run python -m rag.chunk`; inspect one document with
`uv run python -m rag.chunk --show arxiv:1706.03762 --limit 20`.

## Rules

- Size is measured in **`BAAI/bge-small-en-v1.5` tokens** (the embedding model of step 7), not
  words. Limit 350, leaving ≥ 64 of the model's 512 for the "title | section path" header added
  at embedding time. The chunker refuses a limit that leaves no room.
- Chunks never cross a section. Whole paragraphs are packed while they fit.
- A paragraph over the limit is split at sentence ends (abbreviations like "e.g.", "Fig.",
  "et al." and initials like "J." are not sentence ends). Where a chunk boundary falls inside such
  a paragraph, the next chunk repeats the last sentence.
- A sentence over the limit is cut at words; a single over-long "word" at characters.
- Abstract → `abstract` chunks; each caption → its own `caption` chunk; the rest → `body`.
- Every chunk is an exact slice of its section (`section.text[start:end] == chunk.text`); its
  page range comes from the paragraphs it overlaps.

## Result

| Measure | Value |
|---|---|
| Chunks | 400 from 10 papers (body 327, caption 62, abstract 11) |
| Time | 0.4 s for all 10 |
| Tokens per chunk | median 224, mean 204, **max 350** (none over the limit) |
| 0–49 tokens | 46 (12%): 41 captions, 5 body |
| 50–99 / 100–199 / 200–299 / 300–350 | 13% / 19% / 29% / 28% |
| Chunks repeating a sentence from the previous one | 5 (most long paragraphs fit whole) |
| Chunks that are not an exact slice of their section | 0 |

Per paper: Adam 43, word2vec 34, VAE 45, GAN 23, Seq2Seq 30, Distillation 29, DDPM 63,
BatchNorm 36, ResNet 56, Attention 41.

## What checking the output found

1. **Footnotes inside the abstract (fixed).** Attention's "abstract" chunk set was 425 tokens
   (2 chunks): GROBID had filed the author footnotes ("* Equal contribution. Listing order is
   random…", "† Work performed while at Google Brain.") under the abstract. New TEI rule: drop
   abstract paragraphs that start with a footnote mark (`* † ‡ § ¶ ‖`). After: one abstract
   chunk of 241 tokens, matching arXiv's own abstract (240 tokens).
2. **Seq2Seq's abstract is genuinely 359 tokens**, so it is 2 chunks. Left as is.
3. **Adam's misfiled "7.2 TEMPORAL AVERAGING"** (GROBID missed the Conclusion and appendix
   headings; see `seed10_parse_review.md`) became 12 chunks with their own page ranges, pages
   9–15. Size is handled; the section path stays wrong.
4. **Small body chunks** are tiny sections ("5 Training": "This section describes the training
   regime…", 10 tokens) or formula fragments from Adam's proofs (23 and 32 tokens). Kept: chunks
   never merge across sections, and the header added at embedding time gives them context.

## New tool from this step

`uv run python -m rag.ingest tei` re-applies the current TEI rules to the GROBID output saved in
`data/parsed/`, without running GROBID (≈ 0.01 s per paper vs ≈ 1.6 s). Used here for the
footnote rule. It replaces sections, which drops chunks, so run `rag.chunk` after it.
