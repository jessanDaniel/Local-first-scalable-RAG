# Phase 7 — Chroma: what we built ourselves vs. what Chroma gives us

Same document, chunking config, and query used across Phase 6 (manual) and Phase 7 (Chroma):
`company_policy.txt`, chunk_size=100, overlap=10, query="What is the company's refund policy?".

## Comparison

| Concern | What we built manually (Phase 6) | What Chroma gives us |
|---|---|---|
| Similarity math | `cosine_similarity()`, hand-rolled in NumPy | Same math, implemented in C++ (`hnswlib`); also supports L2, dot product |
| Top-K search | `np.argsort` over **all** vectors — exact, O(n) per query | Approximate Nearest Neighbor (HNSW graph) — sub-linear at scale |
| Storage | Python list in memory, recomputed every run | **Persistent** on disk (`chroma_db/`) — embed once, query many times across process restarts |
| Metadata | Attached `document_id`/`position` by hand on a dataclass | Native `metadatas` per vector, with `where={...}` filtering at query time |
| Scaling | Fine at hundreds of chunks; full matrix multiply doesn't scale to millions | Index amortizes lookup cost; designed for large collections |
| Distance space | Cosine similarity, defined by us | Configurable — default is squared L2; must set `metadata={"hnsw:space": "cosine"}` to get cosine-scale scores back |
| What stayed the same | — | Chroma is **only** the storage + search layer. Embedding model choice, chunking, and query embedding are still entirely ours. Nothing about the RAG pipeline's earlier stages is hidden. |

## Key experimental results

1. **Default (L2) collection** (`experiments/learning/chroma_experiment.py`): ranking matched Phase 6 exactly
   (`company_policy_0` → `_5` → `_2`), but scores were L2 distances (0.8444 / 0.9757 / 1.1442, lower
   = closer) — different scale and direction from Phase 6's cosine scores.
2. **Norm check** (`experiments/learning/check_embedding_norms.py`): all `company_policy` chunk embeddings
   have `norm == 1.0000` regardless of word count (100 vs. 23 words). Traced to
   `all-MiniLM-L6-v2`'s `modules.json`: `Transformer → Pooling → 2_Normalize` — normalization is
   baked into the model pipeline, independent of the `normalize_embeddings` flag on `.encode()`.
3. **Cosine-space collection** (`experiments/learning/chroma_cosine_experiment.py`, `hnsw:space: cosine`):
   `1 - distance` reproduced Phase 6's cosine scores **exactly** — 0.5778 / 0.5121 / 0.4279.

## Conclusion

Because this embedding model always emits unit vectors, `L2² = 2 - 2·cos_sim` always holds here, so
default-L2 and cosine ranking are **structurally guaranteed** to agree for any dataset embedded
with `all-MiniLM-L6-v2` — this was not a coincidence specific to `company_policy`. The distance
space setting matters for *score interpretability* (getting cosine-scale numbers vs. raw L2
distances), not for *ranking correctness*, with this particular model. A model/pipeline without a
baked-in `Normalize` module would not carry this guarantee — the general principle (L2 ≠ cosine for
non-unit vectors) still holds, it just isn't a live risk here.

Phase 7 is done. Next: Phase 8 — decouple `retrieve(query, k)` from prompts/providers.
