# Stage 1 step 8 — answers with checked citations (10 papers, qwen2.5:7b-instruct)

2026-11-19. Command:

```bash
uv run python -m rag.ask "why do transformers divide attention scores by the square root of the dimension?" --explain
uv run python experiments/stage1/answer_smoke.py --prompt-variant v1|v2   # -> answer_smoke_v1/v2.jsonl
```

## What was built

- `rag/answer/sources.py`: top 10 hits → merge hits that overlap or touch within a section →
  numbered sources in rank order, within 3,000 prompt tokens (counted with the Qwen2.5 tokenizer)
  and at most 8. Labels like `[2] Attention Is All You Need (arxiv:1706.03762) §3.2.1 …, p.4`,
  links like `https://arxiv.org/pdf/1706.03762#page=4`.
- `rag/answer/prompt.py`: system prompt with citation rules (v2, below) + sources + question.
- `rag/answer/citations.py`: parses `[2]`, `[1][3]`, `[1, 3]`, `[2-4]`; flags numbers with no
  source, answers with no citation; counts cited sentences.
- `rag/answer/answerer.py`: refuses before calling the model if prompt + answer would exceed the
  context window (`ContextWindowError`); warns when the answer hits the token limit.
- LLM layer: Ollama now always sends `num_ctx` (8192) and `num_predict` (600). Checked with
  `ollama ps`: model on **100% GPU**, context **8192**.
- Prompt tokens counted before sending vs Ollama's report: 2043 vs 2024 on the first run, so the
  count is slightly conservative, as intended.

## Smoke check (not the evaluation)

13 questions: the 10 step-7 paraphrases (one per paper), the "why do transformers…" wording that
caused a misattributed quote on the first live run, and 2 questions the corpus cannot answer
(LoRA's rank on GPT-3, GPT-3's parameter count). Every answer was read by hand.

| Measure | prompt v1 | prompt v2 (shipped) |
|---|---|---|
| Expected paper among the sources / cited | 11/11 / 11/11 | 11/11 / 11/11 |
| Citations to numbers that do not exist | 0 | 0 |
| Quotes credited to a source that does not contain them | 1 | 0 |
| Sentences carrying a citation | 27/51 (53%) | 25/42 (60%) |
| Answers containing LaTeX | 6/13 | 2/13 |
| Unanswerable questions refused | 2/2 | 2/2 |
| …refusals that still attach citations | 1/2 (5 sources) | 2/2 (1 source; **all 8** on GPT-3) |

Read by hand (v1 and v2 agree unless noted):

- **Correct and cited to the right section:** √d_k (step-7 wording), Adam bias correction,
  distillation, ResNet degradation/residuals, Seq2Seq source reversal, GAN, BatchNorm covariate
  shift, word2vec skip-gram — 8/10.
- **DDPM:** right idea; v1 cited "Broader Impact" and "Appendix D Samples" with 4/5 sentences
  uncited (likely the model's own knowledge); v2 cites 4.4 Interpolation and the conclusion — better.
- **VAE (wrong in both):** the reparameterization-trick chunk is in the prompt as [3] (rank 4),
  but the DDPM chunks ranked above it lead the model to say the sources "do not directly
  address" the question. Retrieval found the answer; the higher-ranked near-distractors derailed
  generation. Same case as step 7's near-miss.
- **"why do transformers divide …" wording:** the answer chunk (§3.2.1) is ranked **8th of 8**
  here (rank 2 with the step-7 wording: "transformers" pulls in general Transformer chunks). v1
  credited a sentence-long quote to §3.2.2, which does not contain it. v2 drops the quotation
  marks but in the smoke run still cited [1][6] for content from [8]: **the misattribution was
  hidden, not fixed.** The 7B model leans on the earliest sources; the reranker (Stage 2) should
  move the answer chunk to the top.

## Two findings that change how to test answers

1. **A valid citation is not a correct citation.** The checker caught nothing in the misattributed
   cases: [6] exists, so it counts as valid. Checking that the cited source supports the sentence
   needs the deferred evaluation (manual labels first, then an automated support check).
2. **Temperature 0 is not fully reproducible.** The same v2 prompt for the "why do transformers…"
   question cited [8] (correct) in 4 out of 4 standalone runs, but [1][6] when run inside the
   smoke script after other questions. Most likely Ollama's reuse of cached prompt work from the
   previous request changes the GPU arithmetic slightly. The eval should run each question more
   than once and report agreement.

## Timing

First-time questions: **6–13 s** to generate (36–258 answer tokens, ~2–3.2k prompt tokens) on
the M4 Pro GPU; cold model load adds ~10 s to the first call. Re-asking identical prompts was
3–4× faster (1.3–5 s), most likely from Ollama's prompt cache, so those numbers are not used.

## Left for later

- Reranking (Stage 2) for the two rank-order failures above.
- A citation-support check and answer-quality labels (evaluation stage).
- Refusals with citations: a prompt rule did not fix it on a 7B model; candidates are a code-side
  rule (drop citations from a sentence that says information is missing) or a stronger model —
  to be tested, not assumed.
