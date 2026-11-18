# Stage 1 step 7 — embeddings + exact FAISS index on the first 10 papers

2026-11-18. Model `BAAI/bge-small-en-v1.5` (384 dims, 512-token limit, built-in normalize layer,
so inner product = cosine). Index `faiss.IndexFlatIP` (exact search). Chunks: 400, chunker v2-350.

Commands:

```bash
uv run python -m rag.index build      # embed (cached) + new build + CURRENT pointer
uv run python -m rag.index list
uv run python -m rag.search "why divide attention scores by the square root of the dimension" --explain
uv run python experiments/stage1/dense_smoke.py   # this note's numbers -> dense_smoke.csv
```

## Design

- Passage input = `"<title> | <section path>\n<chunk text>"`; header trimmed to 60 tokens
  (outer path levels dropped first, then the title's tail). Chunk ≤ 350 + header ≤ 60 + special
  tokens < 512.
- Query input = bge's instruction `"Represent this sentence for searching relevant passages: "` +
  query, added inside `Embedder.embed_query`.
- Every input is token-counted; over 512 raises `InputTooLongError` instead of being truncated.
- Builds are versioned: `data/indexes/dense/<build_id>/` (index, row → chunk_id map, manifest),
  recorded in `index_builds`, selected by `CURRENT`. Loading refuses an index whose model or
  chunk set differs from settings/database (`StaleIndexError`).
- Vectors are cached by sha1(model + exact input text) in `data/embeddings/`: a rebuild only
  embeds chunks whose input changed.

## Numbers

| Measure | Value |
|---|---|
| Embedding, first build | 400 chunks in 2.36 s on MPS (Apple GPU) = 170 chunks/s, incl. warm-up |
| Rebuild with unchanged chunks | 0 embedded, 400 from cache, 0.04 s |
| Index file | 600 KB (= 400 × 384 × 4 bytes) |
| Query time (explain view) | embed query 71 ms, FAISS search 1.5 ms, fetch from SQLite 1.1 ms |
| Model load | ~14 s first time (download), a few s after |
| FAISS vs hand-written NumPy dot product | identical top-10 for all 10 questions; max score difference 1.2e-7 |

## Smoke check: 10 paraphrased questions, one per paper

Not the evaluation. Rank of the best accepted chunk among all 400 (accepted = the answer's
section, or a chunk containing a phrase only the answer has; see `dense_smoke.py`).

| Question (paraphrased, not the paper's words) | A full | B no instruction | C no header |
|---|---|---|---|
| why divide attention scores by the square root of the dimension | 2 | 1 | 1 |
| correcting moving averages of the gradient that start at zero | 1 | 1 | 1 |
| train a small network on the softened outputs of a big model | 1 | 2 | 4 |
| how to backpropagate through random sampling of a latent variable | 4 | 4 | 5 |
| why very deep networks get worse and how learning a residual fixes it | 1 | 2 | 1 |
| generate images by learning to undo noise that was added step by step | 2 | 2 | 2 |
| reversing the order of the words in the input sentence helps the LSTM | 2 | 2 | 2 |
| two networks trained against each other, one making fakes and one spotting them | 2 | 5 | 6 |
| layer inputs keep changing distribution during training, which slows learning | 1 | 1 | 1 |
| predict the surrounding words given the current word | 1 | 2 | 1 |
| **rank 1 / top 3 / top 5** | **5 / 9 / 10** | 3 / 8 / 10 | 5 / 7 / 9 |
| **mean rank** | **1.7** | 2.2 | 2.4 |
| **top hit from the right paper** | **9/10** | 8/10 | 7/10 |

Reading it:

- **The full design is best on every summary line**, and never worse than rank 4. Removing the
  query instruction (B) or the header (C) each cost something: C let a wrong paper take rank 1
  three times. With only 10 questions these are hints, not proof; the deferred evaluation has to
  confirm them.
- **The header is not free:** on the √d_k question it moved the answer from rank 1 to 2. The
  rank-1 chunk (§3.2.2 Multi-Head Attention) shares most of its header with the answer (§3.2.1),
  and the two scored 0.7078 vs 0.7043.
- **The one real miss is a near-distractor:** "backpropagate through random sampling of a latent
  variable" ranks three DDPM chunks (background on latent-variable models and variational bounds,
  scores 0.750 / 0.750 / 0.741) above VAE's "2.4 The reparameterization trick" (0.740). Two
  papers about the same maths; dense similarity cannot separate them by 0.01. This is the kind of
  case a cross-encoder reranker (Stage 2) is meant to fix — a concrete test case for it.
- Several answers sit at rank 2 behind a neighbouring chunk of the same paper. With 6–8 chunks
  going to the LLM (step 8), rank 2 is still in the context.
