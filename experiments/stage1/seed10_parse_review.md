# Stage 1 step 5c — first 10 papers: ingest run and by-hand parse review

2026-11-18. Corpus: `corpus/seed_ai_research.txt` — the first papers of the "AI research" track
on fanout.sh/papers that are on arXiv (the Perceptron, Dropout and AlexNet are not; the next
three on the list, Batch Normalization, ResNet and Attention, took their places). All 10 ids
were checked against arXiv's own titles before downloading.

Command: `uv run python -m rag.ingest corpus/seed_ai_research.txt --report experiments/stage1/ingest_seed10.csv`
(GROBID 0.8.2 on JDK 17 running locally).

## Run numbers

| Measure | First run | Re-parse after rule fixes |
|---|---|---|
| Status | 10 ok, 0 suspect, 0 failed | 10 ok |
| PyMuPDF fallback used | 0 / 10 | 0 / 10 |
| Parse time per paper | median 1.60 s, max 9.90 s (first paper: GROBID cold) | median 1.60 s, max 2.01 s |
| Downloads | 10 PDFs, 16.3 s including the 3 s arXiv throttle | 0 (cached) |
| Wall clock | 45 s | — |

Stored: 10 papers, 188 sections, 58,914 words. Disk: PDFs 19 MB (11 files incl. QLoRA),
parse dumps 1.7 MB, `rag.db` 576 KB. `ingest_seed10.csv` holds the per-paper rows of the re-parse.

**"10 ok" overstates quality.** The quality gates only catch broken parses (no text, garbled
text, too few sections). Reading every section tree by hand found the problems below, none of
which the gates can see.

## What reading the 10 section trees found

| # | Problem | Papers | Cause | Action |
|---|---|---|---|---|
| 1 | Appendix filed under the last body section ("7 Future work > E Monte Carlo EM", "6 Conclusion > A Extended derivations", "4.3 … > A. Object Detection Baselines") | VAE, DDPM, ResNet | GROBID left the appendix in `<body>`; lettered headings have no `n` and our body rule only knew digits | **Fixed:** a lettered heading after ≥ 2 numbered sections switches the rest of the body to `Appendix > …` |
| 2 | Fake sub-sections from non-heading text: "end for" (algorithm box), "scale and shift" (algorithm), "(middle)." (figure), "2.", "Guanaco:" | GAN, BatchNorm, ResNet, VAE, QLoRA | GROBID tags short lines as headings | **Fixed:** a heading that starts lowercase or with punctuation, ends in ":" or ".", has > 12 words, or no word of ≥ 3 letters is dropped; its text joins the previous section |
| 3 | Headings GROBID never detected: Adam's "8 CONCLUSION" and appendix proofs ended up in "7.2 TEMPORAL AVERAGING" (24 paragraphs, 2,049 words); Adam's "2 ALGORITHM" also missing | Adam | GROBID heading detection | **Not fixable here.** Text is kept, only filed under the wrong path. Watch for very large sections. |
| 4 | Missing parent headings ("6.1" with no "6 Results", "3.1" with no "3") | Attention, ResNet, Adam | GROBID | Already handled: paths skip the missing parent |
| 5 | Remaining fake headings that look like real ones: "Truth" ×2, "Method test BLEU score (ntst14)" (table text), "Input-Input Layer5" ×2 (attention-map figure, pages 13–15) | Seq2Seq, Attention | GROBID reads figure/table text as headings | Left as is: they pass any simple rule. Tucked under the right parent, so citations still point at the right section. |
| 6 | "Appendix > And:" | VAE | First heading of the annex, so there is no earlier section to fold it into | Left as is (79 words) |

Re-parse result for the fixed cases: VAE `Appendix > E Monte Carlo EM`, `Appendix > F Full VB >
F.1 Example`; ResNet `Appendix > A Object Detection Baselines > PASCAL VOC` … `C ImageNet
Localization`; DDPM `Appendix > A Extended derivations` … `D Samples`; GAN, BatchNorm, ResNet
lose their fake sub-sections.

## Takeaways for the next steps

- Section paths are good enough to cite, but not perfect. The chunker (step 6) must not depend
  on a section being small: Adam's 2,049-word "7.2" has to be split like any other long section.
- A status of `ok` means "parsed", not "parsed correctly". A useful extra gate later: flag a
  section holding > 25 % of a paper's body words (Adam's 7.2 is 31 %).
- Fallback rate (0/10) says nothing yet; it needs the 200-paper run planned before Stage 2.
