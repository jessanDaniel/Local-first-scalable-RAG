# Stage 2 step 15b — a reranker switch, and the reranker as a vote

2026-11-22. Nothing new added: the same bge model and the same RRF function.

## What was built

- **Reranking is now separate from the retrieval mode.**
  - `RETRIEVAL_MODE` = hybrid \| dense \| bm25 says how candidates are found.
  - `RERANK` (search, eval) and `ASK_RERANK` (ask) = off \| fused \| replace say what happens next.
  - Any command can override with `--rerank`.
- **`off` never imports the reranker module**, so no model load and no ~1 GB of memory. A test
  replaces the class with one that raises, and proves both that "off" doesn't reach it and that
  "fused" does.
- **`RerankingSearcher` wraps any retriever** and has two strategies:
  - **replace:** the reranker's order is final (step 15's behaviour).
  - **fused:** the reranker's order is a third vote. RRF runs over the dense order, the BM25 order
    and the reranker order of the same 100 candidates.
  - With a single retriever underneath, its own order is the other vote.
- `--explain` shows `dense, bm25, fused, rerank, final` ranks per hit. Eval runs record
  `retriever` and `rerank` separately; runs from step 15 (`retriever: rerank`) are read as
  hybrid + replace.
- 5 new tests:
  - replace: the reranker's order is final
  - fused: two agreeing lists outvote the reranker, and the final order equals RRF recomputed
    from the recorded ranks
  - off never loads the model
  - an unknown strategy is refused
  - CLI parsing

## Results (papers, 40 answerable; hr-demo 24/24 at rank 1 for all)

| | Hybrid | + replace | **+ fused** |
|---|---|---|---|
| Hit@1 | 22 | 22 | **26** |
| Hit@5 | 33 | **37** | 35 |
| Hit@10 | 37 | 37 | **39** |
| Hit@20 | 40 | 39 | 40 |
| MRR | 0.675 | 0.695 | **0.755** |
| nDCG@10 | 0.730 | 0.750 | **0.806** |
| search p50 | 9 ms | 1.8 s | 1.8 s |

Replace reproduced step 15's numbers exactly (22 / 37 / 0.695).

Hybrid → fused: 12 questions better, 2 slightly worse (attn_heads 6 → 8, w2v_cbow 11 → 12),
**no rank-1 answer lost**. vae_kl_term, s2s_smt_baseline, kd_specialists and ddpm_score_matching
reach rank 1. vae_reparam, which replace pushed out of the top 20, stays at rank 1. Fused keeps
part of replace's repairs (s2s_bleu 11 → 7, resnet_ensemble_error 15 → 9, bn_gamma_beta 9 → 4)
rather than all of them (replace got 1, 5, 3). That's the trade: the reranker can no longer
overrule two agreeing lists on its own.

By word overlap (Hit@1 / Hit@5):

| | n | Hybrid | Replace | Fused |
|---|---|---|---|---|
| reworded (≤ 0.5) | 13 | 7 / 10 | 7 / 11 | **9 / 11** |
| copied wording (> 0.5) | 27 | 15 / 23 | 15 / **26** | **17** / 24 |

## Decision rule (fixed before the run) → fused becomes `rag ask`'s default

| Condition | Result |
|---|---|
| papers Hit@1 ≥ 22 | 26 ✅ |
| papers MRR ≥ 0.675 | 0.755 ✅ |
| vae_reparam in the top 5 | rank 1 ✅ |
| hr-demo 24/24 at rank 1 | 24/24 ✅ |

`ASK_RERANK=fused`. `RERANK=off` stays the default for `rag search` and `rag eval`, where the
1.8 s is the whole cost.

## The real cost of `rag ask` from the command line

One question ("what does the KL term push the approximate posterior towards"):
- With fused, the answer chunk §2.3 is ranked 1st (dense 4, BM25 5 → rerank 1).
- With `--rerank off`, it's 4th.
- Wall clock: **32 s vs 19 s.**

The difference is mostly **loading bge on every CLI call**: ~8 s to import and load, ~0.5 s of
GPU warm-up on the first score, then ~1.8 s of reranking. Eval runs load once and warm up, so they
only show the 1.8 s. In Stage 3 the API server loads the model once at startup, so only the 1.8 s
remains. Until then, `--rerank off` is the fast path for quick CLI questions.

Whether fused also gives *better answers*, not just better ranks, is step 16's job (answer-level
eval).
