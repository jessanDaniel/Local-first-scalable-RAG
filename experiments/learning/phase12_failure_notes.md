# Phase 12 — Grounding + Failure Experiments

Mandatory phase: three experiments, each explicitly diagnosing **retrieval failure vs. generation
failure**. Scripts: `experiments/learning/phase12_failure_experiments.py` (experiments 1, 2a, 2b, 3 with
`gpt-oss-20b` via Groq), `experiments/learning/phase12_genfail_ollama.py` (experiment 3 retry with
`llama3.2:1b` via Ollama), plus exploration scripts (`phase12_explore_retrieval_failure.py`,
`phase12_confirm_retrieval_failure.py`) used to find a real retrieval-failure case empirically
rather than guessing one.

## A real bug hit while building this (worth keeping as a lesson)

The first version of `phase12_failure_experiments.py` never called `retriever.index_chunks()` for
two of the four runs — their Chroma collections were brand new and empty. Both "I don't know"
results from that run were **meaningless**: the model wasn't refusing because it correctly
determined the context lacked the answer, it was refusing because the context was empty due to a
script bug. Caught only by reading the printed context debug output and noticing it was blank —
another concrete case of "print the exact thing sent to the LLM," this time catching a bug in the
*experiment*, not the pipeline. Fixed by indexing once per collection before querying, guarded
against re-indexing already-populated collections from earlier exploration runs.

## Experiment 1 — Missing information

**Query:** "Does Acme Corp offer any discounts for students or military members?"
**Doc:** `company_policy`, chunk_size=100, overlap=10, k=3. **Model:** `gpt-oss-20b` (Groq).

- Retrieved chunks: Loyalty Program / Price Matching / Order Cancellation / Refund / Exchange
  sections — verified via context debug that none mention student or military discounts.
- **Result: correctly refused.** Answer: "I don't know."
- **Diagnosis: neither a retrieval nor a generation failure.** The document genuinely doesn't
  contain this information, retrieval correctly returned the most-similar chunks anyway (as it
  must — there's no "no good match" signal in cosine top-K), and generation correctly recognized
  the context didn't answer the question rather than fabricating a plausible-sounding discount
  policy. This is the system working as intended, not a failure to explain.

## Experiment 2 — Retrieval failure

**Query:** "How many consecutive late arrivals trigger a review?"
**Doc:** `employee_handbook`, chunk_size=150, overlap=15. **Model:** `gpt-oss-20b` (Groq).
**Ground truth:** Section 2 states "three late arrivals within a rolling 30-day window."

Found empirically (not guessed) by trying 8 candidate queries and checking which chunk actually
contained the answer at various `k`. Scores at retrieval time:

| Rank | Chunk | Score | Content |
|---|---|---|---|
| 1 | `employee_handbook_9` | 0.3139 | Support ticket P4 acknowledgment times — **irrelevant** |
| 2 | `employee_handbook_1` | 0.3029 | Contains "three late arrivals..." — **the actual answer** |
| 3 | `employee_handbook_10` | 0.2905 | Promotion packet requirements — irrelevant |

**2a — k=1 (deliberately narrow):** only the irrelevant P4 chunk is retrieved. Context contains
zero information about attendance. **Result: correctly refused** — "I don't know." The generation
step behaved correctly *given what it was handed*; it did not hallucinate an answer from an
unrelated support-ticket chunk.

**2b — k=3 (same query):** the correct chunk (`employee_handbook_1`) is now included. **Result:
correct answer** — "Three consecutive late arrivals trigger a review."

- **Diagnosis: retrieval failure at k=1, not a generation failure.** The relevant chunk exists in
  the vector store and is ranked a close second (0.3029 vs. 0.3139 — a 0.011 margin), but "close
  second" is worthless at k=1. This is the same "highest similarity ≠ always the right chunk"
  lesson from Phase 5/6, now shown to have a real, measurable downstream consequence: at k=1 the
  system fails entirely; at k=3 it recovers completely, using the *exact same* retrieval scores and
  the *exact same* model. Nothing about generation changed between 2a and 2b — proof that the
  failure lived entirely in retrieval, not generation.
- Also worth noting: the model's behavior in 2a (correctly refusing rather than hallucinating from
  the wrong chunk) is a *good* sign for generation robustness — the failure didn't compound.

## Experiment 3 — Generation failure

**Query:** "How many loyalty points would I need to redeem for a $25 discount?"
**Doc:** `company_policy`, chunk_size=100, overlap=10, k=3. **Ground truth in context:** "100
points per $5 discount" → (25/5) × 100 = **500 points**.

Retrieved context (same for both models below, confirmed via context debug): Loyalty Program /
Price Matching / Order Cancellation / Refund / Exchange sections. The Loyalty Program chunk
containing the rate is present and correctly retrieved — **retrieval succeeded** in both runs
below.

- **`gpt-oss-20b` (Groq):** "You would need **500 loyalty points**..." — **correct.** First attempt
  at constructing a generation failure did not actually produce one; this model handled the simple
  proportional-reasoning arithmetic fine. Reported honestly rather than forcing a failure that
  didn't happen — the project rule is to test hypotheses, not assume them.
- **`llama3.2:1b` (Ollama), same exact context:** "I don't know. ... it only provides information
  about earning and redeeming loyalty points, and does not mention a discount." — **incorrect.**
  The context literally contains "redeemed for discounts... at a rate of 100 points per $5
  discount" — the model misread or failed to connect the retrieved text to the question, and
  incorrectly claimed the information was absent.
- **Diagnosis: generation failure, isolated by holding context constant and swapping only the
  model.** Retrieval is provably not at fault here — both models received identical, correct
  context. The failure is specific to `llama3.2:1b`'s ability to extract and compute over a
  numeric relationship stated in the text, not to what was retrieved. This is the cleanest possible
  demonstration of "correct chunk retrieved, LLM still gets it wrong."

## Summary — the diagnostic pattern across all three

| Experiment | Retrieval | Generation | Verdict |
|---|---|---|---|
| 1. Missing info | Correct (best-available chunks returned) | Correct (refused, no hallucination) | Not a failure — document genuinely lacks the info |
| 2a. Retrieval failure (k=1) | **Failed** (right chunk excluded) | Correct given its input (refused rather than hallucinating) | Retrieval failure |
| 2b. Same query (k=3) | Fixed (right chunk included) | Correct | Confirms 2a's cause was retrieval, not generation |
| 3. Generation failure | Correct (same context both models) | **Failed** (`llama3.2:1b` misread present info) | Generation failure |

The method that made every diagnosis possible: **hold everything constant except the one variable
being tested** (same query across k=1/k=3 for experiment 2; same context across two models for
experiment 3), and always check the actual printed context/retrieval debug output before
concluding anything about the answer. Every one of the three "expected" experiments only became
trustworthy after fixing the indexing bug and actually looking at what was retrieved — assuming
would have produced two false conclusions (a fake "correctly refused" built on an empty context,
in both experiments 1 and 3, before the bug was found).
