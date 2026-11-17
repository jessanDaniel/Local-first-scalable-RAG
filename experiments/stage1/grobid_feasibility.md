# Stage 1 step 5a — GROBID runs natively on Apple Silicon

2026-11-17. Goal: prove GROBID can parse a research PDF on this Mac without Docker before
building the parser on it. Result: **it works.**

## Setup that worked

| Item | Value |
|---|---|
| GROBID version | **0.8.2** (tag `0.8.2`, cloned to `~/AI-project/tools/grobid`) |
| JDK | Homebrew OpenJDK **17.0.19**, the machine's default `JAVA_HOME`; nothing new installed. (Oracle 11.0.18 also works; first run used it.) |
| Engine | CRF (`wapiti`) for every model in `grobid-home/config/grobid.yaml`; no Python/DeLFT needed |
| Native libs | `grobid-home/lib/mac_arm-64/libwapiti.dylib` and `grobid-home/pdfalto/mac_arm-64/pdfalto`, both arm64 |
| Start | `cd ~/AI-project/tools/grobid && ./gradlew run --no-daemon` (uses the default JDK 17) |
| Ready check | `curl localhost:8070/api/isalive` → `true`; `/api/version` → `0.8.2` |
| Memory | ~1.9 GB resident while running |

## Problems hit on the way

1. **GROBID 0.9.1 needs JDK 21.** Its Gradle build asks for a Java 21 toolchain; this Mac has
   17 and 11. 0.8.2 targets Java 11, so it was used instead of installing a new JDK.
   Upgrading to 0.9.1 means installing `openjdk@21` (Homebrew keg-only, does not change the
   default `java`). Open decision.
2. **0.8.2 builds and runs on JDK 17.** I first assumed Gradle 7.2 would not run on 17 and used
   JDK 11 without testing. Re-tested: `gradlew install` succeeds on 17, and QLoRA parsed on 17
   gives output identical to the JDK 11 run except the timestamp (10.42 s cold, 2.10 s warm).
3. **Spaces in the path break GROBID.** Run from `…/RAG doc/tools/grobid`, Wapiti tried to load
   the model file `"~/AI-project/RAG"` and failed: the native library cuts
   its path at the space. GROBID must live at a path without spaces, hence
   `~/AI-project/tools/grobid`, outside this repo.
4. **Consolidation sends data out.** By default GROBID can look headers and citations up on
   CrossRef. Requests must pass `consolidateHeader=0`, `consolidateCitations=0`,
   `consolidateFunders=0` so nothing leaves the machine.

## One paper: QLoRA (arXiv 2305.14314, cs.LG, 2023)

Request: `POST /api/processFulltextDocument` with consolidation off and
`teiCoordinates=p,head,figure,biblStruct`. Inspect with
`uv run python experiments/stage1/inspect_tei.py data/parsed/2305.14314.tei.xml`.

| Measure | Value |
|---|---|
| Time | 12.19 s cold (models loading), **2.17 s warm**; identical 204,399-byte output both runs |
| Title | "QLORA: Efficient Finetuning of Quantized LLMs" (correct) |
| Authors | 4, correct (Dettmers, Pagnoni, Holtzman, Zettlemoyer) |
| Abstract | 279 words, correct |
| Pages | coordinates on pages 1–26 (`file` reported "3 pages"; it is wrong for this PDF) |
| Body | 24 sections, 117 paragraphs, 8,246 words; every paragraph has page coordinates |
| Figures / tables | 4 figures, 7 tables, with captions |
| References | 73, parsed into `<back>`, separate from body text |
| Back matter | `acknowledgement`, `annex` (appendix), `references` |
| Formulas | 8, as text (e.g. `X Int8 = round 127 absmax(X FP32 ) X FP32 …`), not LaTeX |

## What the parser (step 5b) has to handle

- **Unnumbered headings.** Real subsections like "4-bit NormalFloat Quantization" and "Double
  Quantization" have no `n` attribute. So do false headings: the chat transcripts in §6.1
  produced seven "Guanaco:" sections and one heading made of a sentence ("If you have any
  other questions…"), and "Model" (between §1 and §2) is a figure label. Rule: an unnumbered
  heading becomes a child of the last numbered section (`3 QLORA Finetuning > Double
  Quantization`), never a new top-level section. False headings then end up as noisy
  sub-paths inside the right section rather than as separate sections.
- **Missing parent headings.** §5.1–5.3 exist but no "5" heading was emitted. Section paths
  must not assume every parent is present.
- **Appendix.** The `annex` div holds the appendix (hyperparameters, extra results); index it
  as body text with an "Appendix" path. Drop `references` and `acknowledgement` from the body.
- **Figure in a figure.** One "figure" is a chat example ("User: Please factorize 1833…"),
  not a caption. Captions go in as `caption` chunks only when they start with "Figure"/"Table".
- **Pages.** Take a paragraph's page range from its `coords` (`page,x,y,w,h;…`) — first and
  last page number.
- **Math.** Formulas come out as flattened text. Acceptable for retrieval; BM25 (Stage 2)
  will not match LaTeX symbols in them.
- **Author names.** `persName` has `forename` and `surname` children; join them with a space.

## Step 5b result: the rules applied (`rag/parse/tei.py`)

`uv run python -m rag.parse data/raw/pdf/2305.14314.pdf` on QLoRA: 35 sections (abstract, 24
body sections incl. sub-headings, 9 appendix sections, captions), 9,472 body words, status ok.
"Guanaco:" headings sit under `6 Qualitative Analysis > 6.1 …`; "Model" under `1 Introduction`;
appendix paths come out as `Appendix > B.2 Hyperparameters` (number recovered from the heading
text, since GROBID gives annex heads no `n`). 9 captions kept with labels and pages ("Table 2:
Pile" is a GROBID mis-read, kept); the chat "figure" is dropped. Table *contents* are not
extracted yet, only captions. Pages 17–21 (references) contribute no body text.
