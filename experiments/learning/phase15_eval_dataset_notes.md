# Phase 15 — Evaluation Dataset

30 questions in `experiments/learning/eval_questions.jsonl`, over `company_policy.txt` and
`employee_handbook.txt`. No retrieval run and no generation run in this phase. The only
check is that every supporting quote is real: `uv run python experiments/learning/check_eval_questions.py`
collapses whitespace and requires each quote to occur exactly once in its document. That check
passed (12 direct, 6 multi-step, 6 missing-information, 6 distractor).

## Record shape

| Field | Meaning |
|---|---|
| `id` | Stable name for later metric rows |
| `doc` | File under `documents/` |
| `type` | `direct`, `multi_step`, `missing_information`, or `distractor` |
| `question` | The query to embed |
| `expected_answer` | Short canonical answer. `null` when the document does not contain one |
| `source_quotes` | Exact spans a hit must cover. Empty for missing-information |
| `distractor_quotes` | Tempting wrong spans. Set only on `distractor` rows |

`policy_loyalty_25_and_expiry` is the one derived answer. The document states 1 point per dollar
and 100 points per $5, so a $25 discount is 500 points, and that number is not written in the
file. `answer_kind` is `derived` and the note says so. Every other answerable row is extractive.

## What each type is for

- **Direct** — one fact, one span. Includes the Phase 12–14 anchors: refund window, late arrivals, first-years vacation, loyalty redemption rate.
- **Multi-step** — the answer needs two or more spans. Shipped-order cancellation plus the refund window. Severity 1 update time plus the postmortem deadline. The loyalty row also requires arithmetic.
- **Missing-information** — plausible questions the files do not answer: student or military discount, who pays return shipping, same-day delivery, paid holidays, referral bonus, stock options or RSUs. A correct system refuses.
- **Distractor** — the right span sits next to a similar number. Price-match is 14 days beside the 30-day refund and 45-day exchange. Vacation in years four through seven is 20 days beside 15 and 25. P2 acknowledgment is one hour beside the 15-minute P1 rule. Part-time hours are under 30 beside the 40-hour full-time rule. Individual-contributor notice is two weeks beside the four-week manager rule.

Phase 16 scores retrieval against `source_quotes` (and whether a `distractor_quote` was retrieved instead). Phase 17 scores the answer. This file does not pick a chunk size or a K.
