# Step 18 — quality gate (zero tolerance)

2026-11-23. Nothing new installed. The user chose zero tolerance.

## What was built

- **`src/rag/eval/gate.py`** + **`rag eval gate [--update "reason"]`**. Two deterministic,
  LLM-free checks against the committed **`eval/baseline.json`**:
  1. **Retrieval** with the default retriever (`RETRIEVAL_MODE`, `RERANK`) on papers, hr-demo and
     injection. Fails if any of Hit@1/5/10 or MRR drops, or **any question leaves the top 5**,
     even when the totals hold.
  2. **Guardrail replay:** 213 stored answers (step 16 and step 17 runs, including the injection
     run) go through the current guardrails. Fails if any answer's flags change in either
     direction: a new false flag, or a known catch lost.
- **Improvements pass**, listed as `better`, with a note to record them.
- **`--update "reason"`** rewrites the baseline with the reason, the commit and a history entry,
  so every intended change in quality is documented.
- The gate prints which settings differ from the baseline's, so the cause of a failure is usually
  visible straight away.
- **Skips rather than fails** where the data isn't present (no database: everything skipped; no
  stored answers: replay skipped). A stale index *fails*, because it's actionable (rebuild).
- **`scripts/check.sh` runs the gate by default.** `--fast` skips it; unknown options are refused.
- 9 tests for the comparison and baseline logic.

## Baseline (recorded 2026-11-23, "initial baseline (step 18)")

| Set | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| papers (40) | 22 | 33 | 37 | 0.675 |
| hr-demo (24) | 24 | 24 | 24 | 1.000 |
| injection (9) | 9 | 9 | 9 | 1.000 |

The settings are hybrid retrieval with reranking off, which is the gate's configuration; `rag ask`
additionally reranks (fused). The guardrail replay covers 213 answers.

## Proof: planted regressions all fail, with useful messages

| Planted regression | Gate output |
|---|---|
| Query instruction removed (`EMBEDDING_QUERY_INSTRUCTION=""`) | papers Hit@1 22→21, Hit@5 33→31, MRR 0.675→0.651; adam_adamax_bound and ddpm_fid left the top 5 (10 regressions) |
| BM25 dropped (`RETRIEVAL_MODE=dense`) | papers Hit@1 22→18, Hit@5 33→29, Hit@10 37→32 (10 regressions) |
| Guardrail rounding rule removed (patched in-process) | 5 new flags: s2s_bleu "34.8" ×3, s2s_smt_baseline "33.3" ×2 |
| Unchanged code | passes |

## Cost

`scripts/check.sh` takes **~15 s** (gate 11–12 s: loading the embedding model, ~73 searches,
replaying 213 answers) against ~3 s with `--fast`.

## Limits

- Retrieval is measured without the reranker (it would add ~2.5 min). `rag ask`'s fused reranking
  is covered by step 15b's eval, not by the gate.
- The answer-level LLM eval isn't in the gate: ~10 min, and not exactly reproducible.
- The gate protects this Mac only. The data isn't in git, and there's no CI (no remote).
