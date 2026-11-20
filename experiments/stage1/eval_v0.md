# Eval v0 — retrieval evaluation harness and baseline

2026-11-20. Retrieval only: no answers generated, no LLM.

## What was built

- **Question sets** in `eval/sets/`, gold answers stored as quotes:
  - `papers.jsonl`: 45 questions, 40 answerable. Drafted by Claude, 10 under review in
    `eval/review_papers_sample.md`. Types: 10 paraphrase (the step-7 questions), 18 identifier,
    12 concept, 5 unanswerable; 3 tagged near-distractor.
  - `hr-demo.jsonl`: 30 questions, the Phase 15 set converted (24 answerable).
- **Gold is found when scoring**, from the quotes, so re-chunking never breaks a set.
  `rag eval check` reports every quote as one chunk / several chunks / split / missing.
- **`rag eval run --set NAME [--label L] [--no-filter]`** scores:
  - Hit@1/5/10/20, Recall@10/20, MRR, nDCG@10
  - per type, and per question (first gold rank, word overlap with the gold chunk)
  - mean top-1 score of answerable vs unanswerable questions

  Each run is saved to `data/eval/runs/<set>-<UTC>[-label].json` with its configuration: index
  build and type, model, query instruction, chunker, filter, k, set sha1, git commit.
- **`rag eval compare A B`**: metric deltas, configuration differences, and questions whose
  first gold rank changed, marking those that cross top-1/5/10. Run references are
  `SET:latest~N`; plain `latest` is refused as ambiguous across sets, and runs of different sets
  are refused.
- `rag eval list`. 26 new tests (203 in total).

## Baseline (dense, exact FAISS, bge-small, chunker v2-350, collection filter on)

| | papers (40 answerable) | hr-demo (24 answerable) |
|---|---|---|
| Hit@1 | 18 (45%) | 22 (92%) |
| Hit@5 | 28 (70%) | 24 (100%) |
| Hit@10 | 31 (78%) | 24 (100%) |
| Hit@20 | 37 (92%) | 24 (100%) |
| MRR | 0.569 | 0.951 |
| nDCG@10 | 0.611 | 0.964 |
| mean top-1 score, answerable / unanswerable | 0.775 / 0.668 | 0.753 / 0.662 |

Papers by type, Hit@5: paraphrase 8/10, identifier 11/18, concept 9/12.

Reading it:
- **HR is nearly solved and won't show Stage 2 gains.** There are 29 chunks, each one section,
  in a domain nothing else competes with. The papers set is the one that measures Stage 2.
- **Identifier questions are the weak type** (11/18 at Hit@5). Two miss the top 20 entirely:
  "BLEU 34.81" and "3.57% top-5 error". Exact numbers spread over many similar chunks are what
  dense search can't target, and the test case for BM25.
- **The step-7 smoke check (10/10 in the top 5) was optimistic.** On a wider, mixed set, 30%
  of answers are outside the top 5.
- **Unanswerable questions still score high:** a mean top-1 of 0.67 vs 0.77, and the AlexNet
  question scores 0.73. A similarity threshold alone won't separate "not in the corpus"; the
  answer guardrail needs more than that.

## The harness works

| Check | Result |
|---|---|
| Same configuration twice (`papers` baseline vs rerun) | identical, no question changed rank |
| Query instruction removed (the step-7 ablation) | Hit@1 18 → 16, Hit@5 28 → 24, MRR 0.569 → 0.509; 13 questions changed rank, `compare` lists them |
| The 10 step-7 questions vs step 7's ranks | 7/10 identical; the 3 that differ are gold definitions, below |

Step 7 accepted any chunk of the answer section, or one containing a key phrase. Eval v0 accepts
only the chunk holding the quote. The differences:
- Seq2Seq reversal: 2 → 1, because the gold here is the abstract sentence.
- ResNet degradation: 1 → 10, because the gold is one intro sentence where step 7 accepted all
  of §3.1.
- DDPM reverse process: 2 → >20, because step 7 accepted ~20 "reverse process" chunks.

These are set-design choices, so they're in the review sample.

## Bugs caught while building it

- `latest~1 latest` picked the newest runs across sets. The first "rerun" check compared the HR
  baseline with the papers run. `latest` now has to name a set.
- The word overlap read "None" when the gold chunk wasn't retrieved. It now reads the gold
  chunk from the database.
- The split detector only checked the first and last 4 words, so a split near either end showed
  as "missing". It now matches the longest leading and trailing parts.
- **Parse gap, not fixed:** Adam's Algorithm 1, with its default settings (α = 0.001, …), isn't
  in any chunk; GROBID dropped the algorithm box. No question asks about it.

## Caveats

- Median word overlap is 0.67. The questions were written from the chunks and share their
  wording, so real questions will be harder. The review and later user-written questions should
  bring this down.
- 40 questions: one question moving changes Hit@k by 2.5 points. Counts are reported next to
  percentages for that reason.

## After the user's review (2026-11-20)

The user reviewed 10 questions: 8 kept, 2 rewritten (#1 ddpm_reverse question, #7 kd_soft_targets
quote). `compare` against the baseline: only ddpm_reverse moved (>20 → 5). **The reviewed run
`papers-20261120T080005Z-reviewed` is the baseline for Stage 2:** Hit@1 18/40, Hit@5 29/40,
Hit@10 32/40, Hit@20 38/40, MRR 0.574, nDCG@10 0.620.
