# Step 16 — answer-level evaluation (fused reranking, qwen2.5:7b-instruct)

2026-11-23. Nothing new installed. Only the default configuration was run (hybrid + fused
reranking, 8 sources / 3,000 tokens), by the user's choice. Hybrid vs fused, and 8 vs 5 sources,
are queued as follow-ups.

## What was built

| Command | What it does |
|---|---|
| `rag eval answers --set S --runs N` | Runs `rag ask` on every question N times (in rounds, so repeats aren't back to back). Records answer, sources, citations, gold chunk given / cited, refusal, key-fact match, run-to-run agreement, and timings split into retrieval, reading the prompt, and writing (Ollama now reports prompt, write and load seconds in `Generation`) |
| `rag eval sheet RUN… --per-set papers=11 hr-demo=4` | A varied sample as a Markdown labeling sheet: question, expected answer, **reference passage** (gold quote), answer, and the full text of every cited source. Automatic verdicts are hidden, so they can't anchor the labeller |
| `rag eval labels SHEET` | Parses the filled sheet; reports agreement with each automatic check, and the citations that exist but don't support their claim |
| `rag eval rescore RUN` | Redoes the automatic checks on stored answers without the LLM, after a check is improved |

Code: `src/rag/eval/answers.py`, `src/rag/eval/labels.py`. 24 new tests.

## The runs

| | papers (2 runs) | hr-demo (1 run) |
|---|---|---|
| Answers | 90 | 30 |
| Wall time | 10.4 min | 1.9 min |
| Median per answer | 7.4 s = retrieval 2.0 + reading the prompt 2.4 (~2,700 tokens) + writing 1.0 (42 tokens) | 3.8 s (prompt ~1,300 tokens) |
| Gold chunk among the sources given to the model | 76/80 | 24/24 |
| Gold chunk cited | 64/80 | 24/24 |
| Unanswerable refused (after rescore) | 10/10 | 5/6 |
| Refusals that still cite sources | **10/10** | **5/5** |
| Answers citing a source number that doesn't exist | 5 | 0 |
| Repeat runs: same outcome / same cited chunks / identical text | 44 / 44 / 40 of 45 | — |

Correction to an earlier estimate: reading the prompt takes ~2.4 s, not ~5 s. The earlier figure
came from step 8's total generation time.

## Labels — by Claude, not by the user

The user delegated the labeling. **These 15 labels are Claude's** (marked in the sheet, each with
a reason in its note); a human review stays pending. Sheet:
`eval/labels/answers-fused-2026-11-23.md`.

| Verdict | n |
|---|---|
| correct | 8 |
| partial | 1 (loyalty: expiry right, "250 points" wrong; 100 points per $5 means 500) |
| wrong | 1 (ResNet ensemble error) |
| refused_ok | 5 (all 5 unanswerable questions in the sample) |

**The two failures, each traced to its stage:**
1. `resnet_ensemble_error` (both runs): ResNet's "3.57%" chunk was **not among the 8 sources**
   (a retrieval miss: fused rank 9). The model took the BatchNorm paper's ensemble numbers
   (4.9% / 4.82%) and **called them ResNet's** (a generation failure) instead of saying the
   sources don't cover it. This is the near-distractor problem reaching the answer.
2. `policy_loyalty_25_and_expiry`: the right source was given and cited, and the arithmetic
   failed ($25 → 250 points instead of 500). It's the same question that broke the 1B model in
   Phase 12.

**Citations:** 50 cited sources checked.
- In real (non-refusal) answers: **10 of 12 support their claim, 1 partly, 1 doesn't** (the
  ResNet one).
- Refusals cite almost everything: 32 of 38 refusal citations support nothing. The AlexNet
  refusal also *misdescribes* three sources ("[2]-[8] focus on Adam").

## Which automatic checks to trust (agreement with the 15 labels)

| Check | Agreement | Verdict |
|---|---|---|
| refusal detector | 14/15 → **15/15** after the fix below | trust |
| `key_match` | 7/10 | "match" is trustworthy (no false positives). "No match" is not: 3 correct concept/paraphrase answers were worded differently from the expected answer. So papers' 25 `wrong?` are an upper bound, not a count |
| automatic outcome | 11/15 → 12/15 | the remaining 3 disagreements are all `key_match` false negatives |
| "citation number exists" | passes 33 citations that support nothing | doesn't measure support at all, as step 8 suspected |

## Fixed along the way

- **The sheet cut sources at 1,800 characters.** In entries 1 and 2 the supporting sentence
  (the identity-mapping argument, "|Δt| ≤ α") was in the hidden part. A human labeller would
  have marked correct citations as unsupported. The sheet now shows cited sources in full, with
  a test.
- **The refusal detector missed "the Handbook does not specify …"**, because document names
  weren't counted as source words. It now includes handbook, paper, article, report, policy,
  manual and guide. Both runs were rescored without the LLM.

## Open, feeding later steps

- **Step 17 (guardrails):**
  - Refusals cite sources (15/15).
  - Wrong attribution when the gold chunk is missing (ResNet).
  - Unsupported negative claims: same-day delivery got "Acme Corp does not offer same-day
    delivery", a claim the handbook never makes.
- A human review of the 15 labels (the user, when time allows).
- An automatic citation-support check (an NLI model or LLM judge, calibrated on these labels).
  New things, so ask first.
- Follow-up comparisons with the same harness: hybrid vs fused for `ask`, and 8 vs 5 sources
  (latency).
