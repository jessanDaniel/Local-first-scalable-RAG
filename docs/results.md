# Measured results

The numbers the README (step 31) will cite, one line per result. Eval sets live in `eval/sets/`;
run files in `data/eval/runs/` (`rag eval list`, `rag eval compare A B`). Scripts that produced
the other numbers are in `experiments/`. Older per-step write-ups were removed (2026-11-24); they
are in git history before that date.

## Retrieval (Hit@k / MRR over answerable questions)

| Set (answerable) | Retriever | Hit@1 | Hit@5 | MRR | Notes |
|---|---|---|---|---|---|
| papers (40) | dense, bge-small + query instruction | 18 | 29 | 0.574 | no instruction: Hit@5 −4 |
| papers | BM25 (hand-written, k1 1.2, b 0.75) | 21 | 34 | 0.659 | wins only where the question copies the paper's wording |
| papers | hybrid (dense + BM25, RRF k=60) | 22 | 33 | 0.675 | default for search |
| papers | hybrid + rerank replace (bge-reranker-base) | 22 | 37 | 0.695 | 1.8 s/query; breaks vae_reparam (1 → >20) |
| papers | hybrid + rerank fused (reranker as a 3rd RRF vote) | 26 | 35 | 0.755 | default for ask |
| papers | MiniLM reranker | 21 | 34 | 0.661 | no gain, 0.3 s |
| hr-demo (24) | hybrid | 24 | 24 | 1.000 | 23/24 at rank 1 with per-collection BM25 stats |
| gov-docs (39) | hybrid / fused / dense | 23 / 27 / 26 | 36 / 38 / 38 | 0.734 / 0.797 / 0.782 | 9 federal documents, 43 questions |
| injection (9) | hybrid | 9 | 9 | 1.000 | |

Parsing: the PyMuPDF layout parser equals GROBID on papers (Hit@5 33/33 hybrid, 35/35 fused) at
~0.15 s/paper vs ~1.6 s. Same config reruns give identical retrieval numbers.

## Answers (qwen2.5:7b-instruct via Ollama, temperature 0)

- Step 16, papers 90 answers (2 runs) + hr-demo 30: gold chunk given to the model 76/80, right
  refusals 15/16, median 7.4 s/answer (prompt reading 2.4 s). 15 labels (by Claude, human review
  pending): 8 correct, 1 partial, 1 wrong, 5 right refusals.
- Guardrails + prompt v3: refusals that still cite sources 10 → 0; unanswerable refused 11/11;
  "correct?" 53/64 (was 50).
- Prompt injection (18 relevant attempts): followed 4/18 with prompt v3 (all flagged), 0/18 with
  source sanitising (now on by default). Source-instruction detector: held-out recall 7/12, 0/429
  false positives on real chunks.
- Temperature 0 is not reproducible under Ollama: rerun answer evals more than once.

## Latency and resources (M4 Pro, 24 GB)

- Embedding: 170 chunks/s on MPS; query embedding ~75 ms cold, ~6 ms warm in the server.
- Server: warm /search p50 9 ms (CLI cold 11.8 s); footprint 4.2 GB with every model loaded.
- /ask: median first token 8.7 s, mostly Ollama reading the prompt at ~2.4 ms/token (1.0 ms/token
  in step 16; memory pressure suspected: 7–10 GB swap in use).
- Sync: folder → searchable in 10.7 s; an edited file rescanned in 0.33 s; search during a sync
  p50 8 → 14 ms with one 1.75 s stall (PDF parsing holds the GIL).
- Request log write 0.16 ms. On a full 120 MB log: request list 8 ms, stats over the last 1,000
  requests 18 ms, over every request 621 ms / +105 MB.
