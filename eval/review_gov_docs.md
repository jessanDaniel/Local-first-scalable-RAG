# Review of eval/sets/gov-docs.jsonl (2026-11-24)

**Who reviewed:** an independent Claude agent, asked by the user to do the review in their place
(the user could not do it now). It was given only the question file and the documents, not the
author's opinions, and told to find weaknesses. **A human review is still pending**; this sheet is
what that review should start from.

**Verdicts on the 37 original questions:** keep 13, widen gold 11, rewrite 11, rewrite answer 1,
retag 1, drop 0. All were applied. The set is now 43 questions (39 answerable, 4 unanswerable).

## What the review found, and what changed

1. **The format could not express a rule stated in several documents** (the biggest finding). Gold
   could only live in the question's `doc_id`. Three OPM documents state the same sick-leave rules
   word for word, so a correct hit in the "other" handbook scored as a miss. **Fix:** an optional
   `also_in` list of documents (`src/rag/eval/sets.py`, test in `tests/test_eval.py`). It is used by
   `opm_sick_child_care`, `elder_combined_limit`, `elder_subtract` and `safe_leave_advanced_annual`.
2. **About 10 questions copied the document's wording**, which makes them easier than real
   questions. They were rewritten as scenarios a person would ask, for example "I paid for my own
   meals on a work trip and my employer didn't pay me back. Can I deduct the full cost?" and "I've
   used up all my annual leave and need to move out of an abusive home. Can I get more?".
3. **Expected answers that were incomplete or overstated:**
   - `irs_meals_unreimbursed`: the 80% exception for DOT hours-of-service workers.
   - `safe_leave_third_party`: the supervisor still approves the leave.
   - `nist_password_expiry`: a password change is forced when compromise is suspected.
   - `osha_eye_standard`: "or at least as effective".
   - `irs_tax_home`: the main place of business when there is more than one.
4. **`irs_cruise` was ambiguous.** Two rules answer it: the luxury-water-travel daily cap ("twice
   the highest federal per diem rate") and $2,000 a year for conventions on cruise ships. Both are
   now gold and both are in the expected answer. (The step-21b answer marked "partial" gave the
   second rule, so it was in fact a correct answer.)
5. **Gold too narrow** where long documents repeat a fact: 11 questions gained passages.
6. **`none_military_ppl` was not truly unanswerable.** The childbirth handbook says an employee
   away "on active duty military" is not caring for or bonding with the child, so isn't eligible.
   It is now an answerable conditional question (`military_ppl`).
7. **`opm_childbirth_family_sick_leave` said "In the childbirth and adoption handbook"** only to
   dodge the copies elsewhere. It is now `opm_sick_child_care`, a scenario question with
   cross-document gold.
8. **The autopen question was unnatural**, and "Document approved and signed" was a competing
   answer. It is now `autopen_options`: what decisions the official can choose.
9. **Set-level gaps.** The set had no conditional, comparison, scenario or overview questions, and
   only 4 unanswerable, partly weak. Added:
   - `irs_mileage_vs_actual` and `osha_hardhat_replace` (conditional);
   - `nist_timeout_compare` (comparison, two passages);
   - `safe_leave_police_report` (scenario);
   - `nasa_ai_ideas` (overview);
   - `none_fema_calories` (a near-miss next to the water rule; no calorie figure in any document).

   Still missing, per the reviewer: table lookups (NIST AAL table, IRS tables, OSHA lens table),
   IRS chapters 4–6 (depreciation, reimbursement plans, Form 2106) and more unanswerable near-misses.

**A parser bug the review surfaced:** the cruise sentence contained a running header ("…meetings held
*12 Chapter 1 Travel Publication 463 (2025)* on cruise ships"). Repeated margin text is now removed
line by line, not only as whole blocks (`rag/parse/pdf.py`). IRS now drops 150 header lines.

## For the human review

Start with the questions tagged `review-rewritten` and `review-added`, and with `irs_cruise`,
`military_ppl` and `none_fema_calories`. For each one: would a real person ask it this way? Is the
expected answer right and complete? For the unanswerable ones, is the answer really absent?

Tags in the set: `review-widened` (gold added), `review-rewritten`, `review-added`,
`cross-document` (uses `also_in`), `gold-widened` (widened by the author after reading retrieved
chunks, before the review).
