# Phase 13 — Top-K Experiments

K ∈ {1, 3, 5, 10}, 3 queries across 2 documents, `gpt-oss-20b` (Groq) for generation. Full raw
results in `experiments/learning/topk_experiment.csv`. Script: `experiments/learning/phase13_topk_experiment.py`.

## Results

| Doc | Query | k | chunks (n/total) | words | retrieval hit | answer correct |
|---|---|---|---|---|---|---|
| company_policy | refund days | 1 | 1/6 | 101 | ✅ | ✅ |
| company_policy | refund days | 3 | 3/6 | 226 | ✅ | ✅ |
| company_policy | refund days | 5 | 5/6 | 428 | ✅ | ✅ |
| company_policy | refund days | 10 | 6/6 | 529 | ✅ | ✅ |
| employee_handbook | late arrivals | 1 | 1/16 | 151 | ❌ | ❌ |
| employee_handbook | late arrivals | 3 | 3/16 | 453 | ✅ | ✅ |
| employee_handbook | late arrivals | 5 | 5/16 | 755 | ✅ | ✅ |
| employee_handbook | late arrivals | 10 | 10/16 | 1510 | ✅ | ✅ |
| employee_handbook | vacation days | 1 | 1/16 | 151 | ✅ | ✅ |
| employee_handbook | vacation days | 3 | 3/16 | 453 | ✅ | ✅ |
| employee_handbook | vacation days | 5 | 5/16 | 755 | ✅ | ✅ |
| employee_handbook | vacation days | **10** | 10/16 | 1510 | ✅ | **❌** |

## Findings

1. **Retrieval hit rate improves with k, as expected.** The "late arrivals" query (already known
   from Phase 12 to be a near-miss, 0.3029 vs. 0.3139 score) fails to retrieve the right chunk at
   k=1 and succeeds from k=3 onward. This replicates Phase 12's finding inside a proper sweep
   rather than a one-off.
2. **`k` beyond the document's actual chunk count is a silent no-op, not an error.** `company_policy`
   has only 6 chunks total; `k=10` still only returns 6 (529 words) — Chroma just returns everything
   available. Worth knowing before assuming a bigger `k` number in config always means more context.
3. **The most important finding: more context is not always better, even when the correct chunk is
   included.** The "vacation days" query answers correctly at k=1, 3, and 5 — but at **k=10**, with
   the identical correct chunk still present (retrieval_hit=True), the model's answer degraded to
   *"I'm not sure."* Same query, same document, same model — only `k` changed, and only in the
   direction of *more* information. `employee_handbook` contains several other day-count numbers in
   nearby sections (30-day sick leave bank, 45-day expense reimbursement window, 90-day breaking-change
   deprecation window, 5/10/15/20/25/30/60/90 appearing throughout) — at 1510 words of context, the
   correct "15 days" answer was apparently no longer confidently distinguishable from that noise, and
   the model hedged rather than committing to a (correct) number it had already gotten right with
   less context.
4. **Latency did not scale meaningfully with context size in this experiment** (0.27s–0.74s across
   all rows) — Groq's hosted throughput absorbed the larger prompts without a visible latency
   penalty. This is a hosted-inference-specific result; local CPU inference (Ollama) would likely
   show a clearer latency-vs-context relationship, not tested here.
5. **Operational cost of larger k, observed directly, not just theorized:** running this sweep
   (12 generate calls in quick succession) hit Groq's free-tier rate limit (429) mid-run. Not a
   `k`-specific issue, but a concrete reminder that "more calls / bigger prompts" isn't free even
   on a $0 provider — there's a request-rate ceiling, and retry/backoff (added to the script) is a
   real production concern, not a hypothetical one.

## Answer to the phase's core question

**Is more context always better? No — empirically not.** Retrieval hit rate is monotonically
non-decreasing in k (more chunks can only add coverage, never remove it), but **answer
correctness is not monotonic in k**: it went 1→3→5 correct, then flipped to incorrect at k=10,
purely from context dilution/distraction, with zero change to what was actually retrieved that
mattered. The lesson isn't "always use a small k" either — k=1 failed a different query outright
from under-retrieval. There's an optimal middle range, and it's query/document-dependent, which is
exactly why Phase 14 (chunking × Top-K matrix) and later Phase 15-17 (a real eval dataset with
metrics) exist — one anecdote per k isn't enough to set a default with confidence, the same lesson
already learned in Phase 12 about not trusting a single run.
