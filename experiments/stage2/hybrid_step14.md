# Stage 2 step 14 — hybrid search (dense + BM25, Reciprocal Rank Fusion)

2026-11-20. No new dependency. `rag ask` now retrieves with hybrid by default (user's decision).

## What was built

| File | What it does |
|---|---|
| `src/rag/fusion.py` | `rrf()`: fuses named ranked lists using positions only, `score = Σ 1/(k + rank)` with k = 60; deterministic tie-break; records each chunk's rank in every list |
| `src/rag/search.py` | `HybridSearcher`: dense top 100 + BM25 top 100 (both with the collection filter) → RRF → top k → hits. `last_ranks` records which list each hit came from. Every searcher now has `rank()` (ids and scores only, no SQLite fetch). `make_searcher(conn, mode)` |
| `src/rag/config.py` | `retrieval_mode` = hybrid (the default for `rag ask`, `rag search` and `rag eval run`), `hybrid_candidates` = 100, `rrf_k` = 60 |
| `rag ask --mode`, `rag search --mode`, `rag eval run --retriever` | hybrid \| dense \| bm25. `--explain` shows "dense rank X, BM25 rank Y" per hit |

## Results

papers, 40 answerable questions:

| | Dense | BM25 | **Hybrid** |
|---|---|---|---|
| Hit@1 | 18 | 21 | **22** |
| Hit@5 | 29 | **34** | 33 |
| Hit@10 | 32 | **38** | 37 |
| Hit@20 | 38 | 40 | 40 |
| MRR | 0.574 | 0.659 | **0.675** |
| nDCG@10 | 0.620 | 0.727 | **0.730** |

hr-demo, 24 answerable: hybrid **24/24 at rank 1** (MRR 1.000), against 22/24 for each method
alone.

By word overlap between question and gold chunk (Hit@1 / Hit@5):

| | n | Dense | BM25 | Hybrid |
|---|---|---|---|---|
| reworded (overlap ≤ 0.5) | 13 | 4 / 10 | 3 / 9 | **7 / 10** |
| copied wording (> 0.5) | 27 | 14 / 19 | **18 / 25** | 15 / 23 |

Reading it:
- **Hybrid is clearly better than dense** (Hit@1 +4, Hit@5 +4, MRR +0.10). Only 3 of 40
  questions got worse, each by one place.
- **On reworded questions hybrid beats both methods at rank 1** (7 vs 4 and 3): when the lists
  agree, the answer rises. On HR that agreement takes it to 24/24.
- **Fusion has a cost, the one predicted in the pitch.** When BM25 has the answer high and dense
  doesn't find it, RRF pulls it down:
  - ddpm_fid 1 → 5
  - bn_learning_rate 2 → 5
  - bn_gamma_beta 3 → 9
  - s2s_bleu 5 → 11
  - resnet_ensemble_error 8 → 15

  That's why hybrid trails BM25 alone at Hit@5 on questions that copy the paper's wording.
  Weighting one list, or changing k, could trade this, but tuning on 40 questions isn't
  justified. The reranker (step 15) is the intended fix: it reads the chunks instead of trusting
  either rank.
- Fused scores are tiny (≈0.03) and nearly equal for answerable and unanswerable questions, so
  they can't be used to decide when to refuse. That's for later.

## End-to-end: the two step-8 failures through `rag ask` (qwen2.5:7b-instruct)

| Question | Dense | Hybrid |
|---|---|---|
| "how to backpropagate through random sampling of a latent variable" | the VAE §2.4 chunk ranked 4th behind 3 DDPM chunks; the answer says the sources "do not directly address" it (wrong) | VAE §2.4 **ranked 1st (dense 4, BM25 3)**; correct answer, cites §2.4 |
| "why do transformers divide attention scores by the square root of the dimension?" | §3.2.1 ranked 8th of 8 sources; cited correctly this run (step 8 showed it varies) | §3.2.1 **ranked 1st (dense 8, BM25 2)**; both sentences cited |

Still open: the VAE answer contains LaTeX despite prompt v2 asking for plain text (seen in step
8 too).

## Timing (per query, 429 chunks)

Dense ~70 ms, almost all of it embedding the query. BM25 0.3 ms. Fusion < 1 ms. Building the BM25
index adds ~45 ms at startup. Running the two in parallel would save almost nothing, so it wasn't
done.
