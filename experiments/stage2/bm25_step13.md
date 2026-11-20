# Stage 2 step 13 — BM25, written by hand

2026-11-20. No new dependency. Tantivy was installed by mistake, then removed before this step
on the user's instruction; SQLite FTS5 is the candidate if a library is ever needed. See
"Library or not" below.

## What was built

| File | What it does |
|---|---|
| `src/rag/index/tokenize.py` | Text → terms, the same for chunks and queries. Lowercase + NFKC; compounds kept whole *and* split (`34.81` → `34.81`, `34`, `81`; `GPT-4` → `gpt-4`, `gpt`, `4`); letter/digit splits (`β2` → `β`, `2`); stopwords dropped, numbers kept; **no stemming** (version `t1`) |
| `src/rag/index/bm25.py` | `BM25Index`: an inverted index (term → [(chunk, tf)]), Lucene-style IDF, BM25 scoring with k1=1.2, b=0.75, a collection filter applied while scoring, and `explain()` (each term's contribution) |
| `src/rag/search.py` | `BM25Searcher`: indexes "title \| section path" + chunk text (the same header the embeddings get); `make_searcher(conn, "dense"\|"bm25")`; dense-only libraries load only in dense mode |
| `rag search --mode bm25 [--explain]` | `--explain` prints, per hit, `term: tf × idf → score` |
| `rag eval run --retriever bm25` | Runs record k1, b and the tokenizer version |

The index is built in memory from SQLite on every start (45 ms for 429 chunks), so it can never
be stale. A query takes 0.3 ms, against ~70 ms for dense search, almost all of which is embedding
the query.

## Results (against the reviewed dense baseline)

| | papers dense | papers BM25 | hr-demo dense | hr-demo BM25 |
|---|---|---|---|---|
| Hit@1 | 18/40 | **21/40** | 22/24 | 22/24 |
| Hit@5 | 29/40 | **34/40** | 24/24 | 24/24 |
| Hit@10 | 32/40 | **38/40** | 24/24 | 24/24 |
| Hit@20 | 38/40 | **40/40** | 24/24 | 24/24 |
| MRR | 0.574 | **0.659** | 0.951 | 0.958 |

Papers by type, Hit@5 dense → BM25: identifier 11 → 15, concept 9 → 10, paraphrase 8 → 9.
22 questions changed rank. The biggest gains: s2s_bleu (>20 → 5), bn_gamma_beta (17 → 3),
resnet_ensemble_error (>20 → 8), adam_name and ddpm_fid (13 → 1). The biggest losses:
adam_adamax_bound (3 → 8), ddpm_score_matching (1 → 5), kd_soft_targets (1 → 4).

## Why BM25 "won": the question set favours it

Split by word overlap between question and gold chunk (Hit@5):

| Questions | n | Dense | BM25 |
|---|---|---|---|
| overlap > 0.5 | 27 | 19 | **25** |
| overlap ≤ 0.5 | 13 | **10** | 9 |

All of BM25's lead comes from questions that reuse the paper's wording. Claude wrote them while
reading the chunks (eval v0's caveat). On genuinely reworded questions dense search is still
slightly better. The two fail on different questions, which is the case for fusion (step 14),
not for dropping dense search.

## Two corrections to what was expected

1. **BM25 matches the question's words, not the answer's.** "What BLEU score did … reach?" has
   no "34.81" in it; the number is the answer. BM25 improved s2s_bleu (>20 → 5) because the
   question's other words (ensemble, deep, LSTMs, WMT'14, English, French) are rare and
   specific. Exact tokens help when they are *in the question*: SGVB, β2, ResNet-50.
2. **BM25 alone was expected to score lower than dense overall. It scored higher**, for the
   reason above. A set with more user-written, reworded questions would likely reverse that.

## Library or not

At 429 chunks the hand-written index is enough: 45 ms to build, 0.3 ms per query. A library
becomes worth it when rebuilding in memory on every start is too slow (roughly 10⁵–10⁶ chunks),
or when the index must update incrementally. The candidate then is SQLite FTS5: nothing new to
install, and it lives inside `rag.db`. Its rankings would be checked against this
implementation, the way Phase 7 checked Chroma against Phase 6. Decided with the user, not
assumed.

## Not done (open questions)

- **Stemming:** no question in either set obviously failed for lack of it. Measure before adding
  it.
- **Field weights** (title vs body): one bag of words for now.
- **Tuning k1 and b:** left at the standard values on purpose, so we don't tune to 40 questions.
- **More reworded questions in the papers set** (ideally written by the user), to reduce the
  bias above.
