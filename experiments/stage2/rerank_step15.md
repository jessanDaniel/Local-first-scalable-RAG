# Stage 2 step 15 — cross-encoder reranking

2026-11-20. **No new Python package.** `sentence-transformers`' `CrossEncoder` was already
installed. Two models were downloaded with the user's approval (option a, compare both):
`cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters, 88 MB) and `BAAI/bge-reranker-base`
(278M, 1.1 GB).

## What was built

| File | What it does |
|---|---|
| `src/rag/rerank.py` | `Reranker`: scores (question, "title \| section path\nchunk") pairs in batches on MPS; uses **raw logits** (sentence-transformers applies a sigmoid to bge by default, squashing scores to ~0.001); a pair over 512 tokens drops its header first, and anything still truncated is counted (`last_truncated`) |
| `src/rag/search.py` | `RerankingSearcher`: hybrid's top 100 → reranker → best 20. If the reranker raises or takes longer than `rerank_timeout_seconds` (10 s), it falls back to the fused order and says why. `--explain` shows dense / BM25 / fused / rerank ranks per hit. `HybridSearcher.rank()` returns the fused list without fetching from SQLite |
| `src/rag/store/db.py` | `passages()`: title, section path and text for a list of chunk ids |
| settings | `rerank_model` (now bge), `rerank_candidates` 100, `rerank_keep` 20, `rerank_batch_size` 32, `rerank_timeout_seconds` 10 |
| eval | Every run now records per-question search latency (p50/p95/max) and the number of fallbacks; the runner warms up before timing; `compare` shows latency deltas |

## Checks before trusting any number

- **The first sanity test looked broken.** A relevant and an irrelevant pair scored almost the
  same (MiniLM −11.42 vs −11.38). The causes:
  - MiniLM's scores matched the plain `transformers` reference on CPU and MPS, so the library
    and the GPU were fine.
  - My test sentence simply lacked the query's words.
  - bge's 0.001 vs 0.000 was the sigmoid hiding raw scores of −7.1 vs −10.2.

  On real chunks both models separate gold from unrelated chunks by 6–19 points.
- **bge's strangest pick was rescored outside the searcher.** For the √dₖ question it put a
  Seq2Seq "long sentences" chunk first. Scoring the pair directly gives the same scores as the
  run (−1.42 vs −1.97 for the real answer), with no truncation (152 and 424 tokens). This is the
  model's own error, not a pipeline bug.

## Results

papers, 40 answerable questions (hybrid rerun with timing):

| | Hybrid | + MiniLM | + bge |
|---|---|---|---|
| Hit@1 | **22** | 21 | **22** |
| Hit@5 | 33 | 34 | **37** |
| Hit@10 | 37 | 37 | 37 |
| Hit@20 | **40** | 39 | 39 |
| MRR | 0.675 | 0.661 | **0.695** |
| nDCG@10 | 0.730 | — | **0.750** |
| search latency p50 / p95 | **9 / 12 ms** | 300 / 362 ms | 1772 / 1876 ms |

hr-demo: 24/24 at rank 1 for all three.

**bge repairs what fusion pulled down.** These are the step-14 cases where BM25 had the answer
and dense didn't:

| Question | Hybrid | + bge |
|---|---|---|
| s2s_bleu | 11 | **1** |
| ddpm_fid | 5 | **1** |
| resnet_ensemble_error | 15 | **5** |
| bn_gamma_beta | 9 | **3** |

Also vae_sgvb 7 → 2, resnet_degradation 9 → 4, adam_adamax_bound 4 → 1, vae_kl_term 4 → 1.

**But Hit@1 doesn't move, because bge also demotes 7 rank-1 answers.** Reading the chunk it put
first instead:

| Question | bge's top-1 | Verdict |
|---|---|---|
| adam_bias_correction | §3 Initialization bias correction | also a correct answer (gold artifact) |
| bn_covariate_shift | the abstract, which states it almost verbatim | also correct (gold artifact) |
| resnet_deepest | the abstract ("depth of up to 152 layers") | also correct (gold artifact) |
| gan_adversarial | a BatchNorm chunk about "the two networks" | **wrong** |
| attn_sqrt_dk | a Seq2Seq "long sentences" chunk | **wrong** |
| attn_bleu_de, (others) | — | not inspected |

**vae_reparam, the near-distractor case, breaks: rank 1 → outside the top 20.** bge prefers VAE
§2.1 and DDPM chunks to §2.4. The pre-check had already shown bge rating a DDPM chunk slightly
above the VAE answer (−4.00 vs −4.22).

MiniLM (trained on web search) is a wash: 20 questions move in both directions, and the net is
slightly worse than hybrid.

## Decision

- **`retrieval_mode` stays hybrid.** bge's gain (Hit@5 +4, MRR +0.02) doesn't justify 1.8 s per
  question plus new rank-1 errors on this set.
- **`--mode rerank` is available**, using bge (`rerank_model`). Its fallback is tested.
- Both models stay in the Hugging Face cache (88 MB + 1.1 GB) until the user decides. Deleting
  them is the user's call.

## What this step taught

1. **A reranker isn't automatically better.** A general-purpose cross-encoder can be worse than
   the fusion of two simpler signals on a specialised corpus. Measured, not assumed.
2. **Strict single-gold labels understate rerankers** (3 of 7 "demotions" were also-correct
   chunks). Any further reranker work needs multi-gold labels, or answer-level evaluation, to be
   judged fairly.
3. **Latency estimates were optimistic.** bge took 1.8 s against an estimate of 300–800 ms.
   Measuring on the machine is the only reliable number.

## Possible next experiments (not done)

- Fewer candidates (e.g. 30 instead of 100) cuts bge's cost about 3×. It's a latency trade-off,
  not a quality fix.
- A stronger or domain-matched reranker (bge-reranker-v2-m3, 568M) would mean a new download, so
  ask first.
- Combining the reranker score with the fused rank instead of replacing it.
- More gold quotes for questions with several valid answer chunks.
