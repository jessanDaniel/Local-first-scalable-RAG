# Phase 14 — Chunking × Top-K Matrix

Same three Phase 13 queries, swept across chunk size, overlap, and K. Retrieval for every
cell is in `evaluations/phase14_chunk_topk_matrix.csv` (48 rows, no LLM). Generation for the
nine cells where rank changed is in `evaluations/phase14_generation.csv` (`gpt-oss-20b` via
Groq). Scripts: `phase14_chunk_topk_matrix.py`, `phase14_generation.py`.

Grid (word counts): policy `{100, 200} × overlap {0, ~10%}`; handbook `{150, 300} × overlap
{0, ~10%}`. K ∈ {1, 3, 5, 10}. The cuts `(100, 10)` and `(150, 15)` are the Phase 13 baselines.
Policy was not sent to the LLM again: every cut already had the refund sentence at rank 1.

`needle_in_index` was true on all 48 rows. None of the three needle phrases was split into
oblivion. Rank still moved, because a phrase can sit intact inside a chunk that is a weak match,
or a sentence can be cut so the highest-ranked chunk holds only the other half.

## Retrieval

Rank of the needle (blank = not in the top K). Hit rate only goes up as K grows; the rank
itself is a property of the cut.

| Query | Cut | Chunks | Rank | First K that hits |
|---|---|---|---|---|
| Refund days | 100/0, 100/10, 200/0, 200/20 | 5, 6, 3, 3 | 1 | 1 |
| Late arrivals | 150/0 | 15 | 1 | 1 |
| Late arrivals | 150/15 (Phase 13) | 16 | 2 | 3 |
| Late arrivals | 300/0 | 8 | 6 | 10 |
| Late arrivals | 300/30 | 8 | 7 | 10 |
| Vacation days | 150/0 | 15 | 2 | 3 |
| Vacation days | 150/15 (Phase 13) | 16 | 1 | 1 |
| Vacation days | 300/0 | 8 | 1 | 1 |
| Vacation days | 300/30 | 8 | 5 | 5 |

Overlap and larger chunks do not help both handbook queries. Dropping overlap on the size-150
cut moves late arrivals from rank 2 to rank 1, and moves vacation days from rank 1 to rank 2.
Size 300 buries late arrivals at rank 6–7. Adding overlap 30 on that cut buries vacation days
at rank 5, where size 300 with no overlap had it at rank 1.

K larger than the chunk count still clamps. Policy at size 200 has 3 chunks, so K=5 and K=10
return the same 3.

## Generation

Nine calls. `answer_correct` is a whole-word check (`three`, `15`). The label below uses the
answer text.

| Query | Cut | K | Hit | Answer | Label |
|---|---|---|---|---|---|
| Late arrivals | 150/0 | 1 | yes | "Three consecutive late arrivals trigger a review." | Both worked |
| Late arrivals | 300/0 | 5 | no | "I don't know." | Retrieval failure |
| Late arrivals | 300/0 | 10 | yes | "Three late arrivals within a rolling 30-day period…" | Both worked |
| Vacation days | 150/0 | 1 | no | "I don't know." | Retrieval failure |
| Vacation days | 150/0 | 10 | yes | "I don't know." | Generation failure |
| Vacation days | 300/0 | 1 | yes | "15 days of paid vacation…" | Both worked |
| Vacation days | 300/0 | 10 | yes | "15 days of paid vacation…" | Both worked |
| Vacation days | 300/30 | 3 | no | "I don't know." | Retrieval failure |
| Vacation days | 300/30 | 10 | yes | "15 days of paid vacation…" | Both worked |

Late arrivals followed retrieval on every generated cell. The size-300 miss at K=5 became a
correct answer at K=10 with no other change.

Vacation days at 150/0, K=10 is the generation failure. The source sentence is:

> Employees accrue 15 days of paid vacation in years one through three of service, 20 days in years four through seven, and 25 days thereafter.

That sentence is split on the size-150, overlap-0 boundary. `build_context` emits chunks in
rank order, so the prompt shows:

- Rank 1 (`employee_handbook_5`, score 0.4508): starts at "20 days in years four through seven, and 25 days thereafter," then other day counts.
- Rank 2 (`employee_handbook_4`, score 0.4274): ends with "Employees accrue 15 days of paid vacation in years one through three of service,"

The first-year fact is in the prompt. It is a trailing fragment, and the higher-scored chunk
is the other half of the same sentence. The template tells the model to say it does not know
when the context does not contain the answer. The model said "I don't know."

The size-300, overlap-0 cut keeps that whole clause in one chunk, and that chunk is rank 1.
The same model answers "15 days" at K=1 (337 prompt words) and at K=10 (2219 prompt words).
A longer prompt was fine once the answering sentence was intact and ranked first. The Phase 13
"K=10 hurts" result was for the 150/15 cut; this matrix shows that outcome depends on the cut.

## What this phase settles

There is no single best chunk size, overlap, or K across these queries. A change that promotes
one fact demotes the other. Answer correctness tracks retrieval except when context construction
hands the model a split sentence under a refusal instruction. Choosing a default from one query
is not supported. That is the reason for a real question set in Phase 15.
