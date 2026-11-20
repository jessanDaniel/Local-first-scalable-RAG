# Review: 10 of the 45 paper questions

Drafted by Claude; for the user to accept, rewrite or reject. The 10 were picked because they
need a human decision: they miss badly, have a strict gold answer, copy the paper's wording, use
a fragment quote, or have no answer in the corpus. They are not a random sample.

For each one, judge:
- **Fair question?** Would a real user ask it this way?
- **Right gold?** Is the quoted sentence *the* answer, or only one of several chunks that
  answer it?
- **Copied wording?** "Overlap" is the share of the question's content words found in the gold
  chunk. 1.0 means every word is copied.

Ranks are from the baseline (`papers-20261120T074426Z-baseline`), rank of the gold chunk among
the 400 paper chunks. Mark each line **keep / rewrite / reject** (or edit
`eval/sets/papers.jsonl` directly).

| # | id | Rank | Overlap | Decision |
|---|---|---|---|---|
| 1 | ddpm_reverse | >20 | 0.14 | |
| 2 | resnet_degradation | 10 | 0.50 | |
| 3 | s2s_bleu | >20 | 0.82 | |
| 4 | resnet_ensemble_error | >20 | 0.67 | |
| 5 | bn_learning_rate | 13 | 0.83 | |
| 6 | adam_name | 13 | 0.67 | |
| 7 | kd_soft_targets | 1 | 0.43 | |
| 8 | gan_nonsaturating | 3 | 0.71 | |
| 9 | vae_kl_term | 4 | 0.45 | |
| 10 | none_alexnet_optimizer | — | — | |

---

**1. ddpm_reverse** (paraphrase, from step 7)

- **Question:** generate images by learning to undo noise that was added step by step
- **Gold quote** (§2 Background): "…is called the reverse process, and it is defined as a Markov
  chain with learned Gaussian transitions"
- **Note:** a broad "what is this paper about" question. Step 7 accepted any of ~20 chunks
  mentioning the reverse process and ranked one 2nd. With a single strict gold it falls out of
  the top 20.
- **My suggestion:** add the abstract's opening sentence as a second gold quote. Otherwise the
  question measures which one of 20 valid chunks wins.

**2. resnet_degradation** (paraphrase, from step 7)

- **Question:** why very deep networks get worse and how learning a residual fixes it
- **Gold quote** (§1 Introduction): "we address the degradation problem by introducing a deep
  residual learning framework"
- **Note:** step 7 had rank 1 by also accepting §3.1. Two parts are asked (why worse, how
  fixed), and they're answered in different chunks.
- **My suggestion:** add the §3.1 residual-mapping sentence as a second gold quote.

**3. s2s_bleu** (identifier)

- **Question:** What BLEU score did the ensemble of deep LSTMs reach on WMT'14 English to French?
- **Gold quote** (§1): "we obtained a BLEU score of 34.81 by directly extracting translations
  from an ensemble of 5 deep LSTMs"
- **Note:** a fair, typical lookup, and the miss is real. The 34.8 / 34.81 / 33.30 numbers are
  spread over many similar chunks, and dense search can't target "34.81". This is the case BM25
  is for.
- **My suggestion:** keep.

**4. resnet_ensemble_error** (identifier)

- **Question:** What top-5 error did the ResNet ensemble reach on the ImageNet test set?
- **Gold quote** (§1): "Our ensemble has 3.57% top-5 error on the ImageNet test set"
- **Note:** the same number appears in the abstract and §4.1 too, but only the §1 sentence is
  gold, so the abstract (which is retrieved) doesn't count.
- **My suggestion:** add the abstract and §4.1 sentences as gold quotes. Answering from any of
  them is correct.

**5. bn_learning_rate** (concept)

- **Question:** How does batch normalization change which learning rates can be used?
- **Gold quote** (§1): "This allows us to use much higher learning rates without the risk of
  divergence."
- **Note:** overlap 0.83, so the wording is close to the paper's, yet it ranks 13th. The
  abstract says nearly the same ("allows us to use much higher learning rates").
- **My suggestion:** add the abstract sentence as a gold quote.

**6. adam_name** (identifier)

- **Question:** What does the name Adam stand for?
- **Gold quote** (§1): "the name Adam is derived from adaptive moment estimation"
- **Note:** rank 13, with the top-1 score only 0.53. "Adam" alone is a weak query for an
  embedding model. A fair question.
- **My suggestion:** keep.

**7. kd_soft_targets** (paraphrase, from step 7)

- **Question:** train a small network on the softened outputs of a big model
- **Gold quote** (§1): "class probabilities produced by the cumbersome model as"
- **Note:** the quote is a fragment, because the full sentence contains curly quote marks that
  are awkward to match. It still identifies exactly one chunk.
- **My suggestion:** keep, or accept a longer quote if you prefer readable gold.

**8. gan_nonsaturating** (concept)

- **Question:** Why is the generator trained to maximize log D(G(z)) rather than minimize
  log(1 - D(G(z)))?
- **Gold quote** (§3): "provides much stronger gradients early in learning"
- **Note:** the question gives away the formulas, which makes it easier than a real user's
  question would be.
- **My suggestion:** rewrite as "Why does the GAN paper change the generator's objective in
  practice?"

**9. vae_kl_term** (concept, near-distractor)

- **Question:** In auto-encoding variational Bayes, what does the KL divergence term push the
  approximate posterior towards?
- **Gold quote** (§2.3): "encouraging the approximate posterior to be close to the prior"
- **Note:** DDPM has KL terms too, but the VAE chunk still ranks 4th. A fair near-distractor.
- **My suggestion:** keep.

**10. none_alexnet_optimizer** (unanswerable)

- **Question:** Which optimizer and learning rate did AlexNet use?
- **Gold:** none (AlexNet isn't in the corpus yet)
- **Note:** top-1 is an Attention paper chunk with score 0.73, above some answerable questions'
  scores. This becomes answerable once you add AlexNet to `data/raw/manual/`.
- **My suggestion:** keep for now, and move it when AlexNet is ingested.

---

**Overall caveat.** Median overlap is 0.67 (paraphrase 0.50, identifier 0.67, concept 0.65).
These questions were written by someone who had read the chunks, and it shows. Expect real
users' questions to be harder than this set.
