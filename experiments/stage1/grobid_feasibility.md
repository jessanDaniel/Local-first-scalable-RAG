# Stage 1 step 5a — GROBID runs natively on Apple Silicon

2026-11-17. Goal: prove GROBID can parse a research PDF on this Mac without Docker before
building the parser on it. Result: **it works.**

## Setup that worked

| Item | Value |
|---|---|
| GROBID version | **0.8.2** (tag `0.8.2`, cloned to `~/AI-project/tools/grobid`) |
| JDK | Oracle **11.0.18** (`/Library/Java/JavaVirtualMachines/jdk-11.jdk`), already installed; nothing new installed |
| Engine | CRF (`wapiti`) for every model in `grobid-home/config/grobid.yaml`; no Python/DeLFT needed |
| Native libs | `grobid-home/lib/mac_arm-64/libwapiti.dylib` and `grobid-home/pdfalto/mac_arm-64/pdfalto`, both arm64 |
| Start | `cd ~/AI-project/tools/grobid && JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-11.jdk/Contents/Home ./gradlew run --no-daemon` |
| Ready check | `curl localhost:8070/api/isalive` → `true`; `/api/version` → `0.8.2` |
| Memory | ~1.9 GB resident while running |

## Problems hit on the way

1. **GROBID 0.9.1 needs JDK 21.** Its Gradle build asks for a Java 21 toolchain; this Mac has
   17 and 11. 0.8.2 targets Java 11, so it was used instead of installing a new JDK.
   Upgrading to 0.9.1 means installing `openjdk@21` (Homebrew keg-only, does not change the
   default `java`). Open decision.
2. **0.8.2 builds with Gradle 7.2, which needs JDK 11 here** (7.2 predates official JDK 17
   support).
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
