"""Stage 1 step 8 smoke check: end-to-end answers with citations on qwen2.5:7b-instruct.

Not the evaluation (still deferred). 10 answerable questions (the step 7 set, one per
paper) + 2 the corpus cannot answer. Per answer: the answer text, citation check, whether
the expected paper is cited, tokens, timings, and an attribution probe:

    for every quoted span ("...", >= 6 words) and every cited sentence, which source's
    words does it overlap most, and is that the source it was credited to?

The probe is a word-overlap heuristic for diagnosis, not a groundedness metric.

    uv run python experiments/stage1/answer_smoke.py [--prompt-variant NAME]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rag.answer import prompt as prompt_module
from rag.answer.answerer import Answerer
from rag.answer.citations import MARKER, marker_numbers
from rag.chunk.chunker import sentence_spans
from rag.chunk.tokens import HFTokenCounter
from rag.config import get_settings
from rag.embed.embedder import Embedder
from rag.index.dense import load_current
from rag.llm.factory import create_llm
from rag.search import DenseSearcher
from rag.store import db

sys.path.insert(0, str(Path(__file__).parent))
from dense_smoke import QUESTIONS

# The wording that made v1 credit a quote to the wrong source in the first live run.
EXTRA = [("why do transformers divide attention scores by the square root of the dimension?", "arxiv:1706.03762")]

PROMPT_V2 = """You answer questions using only the numbered sources you are given.

Rules:
1. Every sentence that states a fact must end with the number of the source it comes from, in square brackets, like [2] or [1][3]. If you cannot point to a source for a sentence, leave the sentence out.
2. Cite only the source numbers given to you, and only a source whose text says what the sentence says.
3. Do not use outside knowledge, even if you know the answer.
4. Use quotation marks only for words copied exactly from the source you cite.
5. If the sources answer only part of the question, answer that part, then say plainly which part they do not cover. A sentence saying something is missing gets no citation.
6. If the sources do not answer the question at all, say so in one sentence, naming what is missing, with no citation.
7. Write math in plain text, for example sqrt(d_k), not LaTeX.
8. Be concise: a short paragraph unless the question asks for more."""

# v1: the first step 8 prompt, kept verbatim so the comparison can be rerun.
PROMPT_V1 = """You answer questions using only the numbered sources you are given.

Rules:
1. After every sentence that uses a source, cite it with its number in square brackets, like [2] or [1][3].
2. Cite only the source numbers given to you. Never invent a source or a number.
3. Do not use outside knowledge. If a fact is not in the sources, do not state it.
4. If the sources answer only part of the question, answer that part and then say plainly which part the sources do not cover.
5. If the sources do not answer the question at all, say so in one sentence and name what is missing.
6. Be concise: a short paragraph unless the question asks for more."""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
assert PROMPT_V2 == prompt_module.SYSTEM_PROMPT, "v2 is the shipped prompt"

UNANSWERABLE = [
    "What rank r did the LoRA paper use when adapting GPT-3?",
    "How many parameters does GPT-3 have?",
]
STOPWORDS = {"the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "are", "was", "were", "be", "by", "with", "on", "as", "that", "this", "it", "its", "at", "from", "we", "our", "which", "can", "not", "than", "these", "those", "their", "they", "has", "have", "each", "into", "such"}
QUOTE = re.compile(r"[\"“]([^\"”]{20,})[\"”]")


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def best_source(text: str, sources) -> tuple[int, float]:
    target = words(text)
    if not target:
        return 0, 0.0
    scores = [(s.n, len(target & words(s.text)) / len(target)) for s in sources]
    return max(scores, key=lambda x: x[1])


def attribution(answer) -> dict:
    by_n = {s.n: s for s in answer.sources}
    text = answer.text
    quotes = []
    for match in QUOTE.finditer(text):
        quote = match.group(1)
        if len(quote.split()) < 6:
            continue
        nearby = text[max(0, match.start() - 40):match.end() + 40]
        credited = sorted({n for m in MARKER.finditer(nearby) for n in marker_numbers(m.group(1))})
        best, overlap = best_source(quote, answer.sources)
        credited_overlap = max((len(words(quote) & words(by_n[n].text)) / max(1, len(words(quote)))
                                for n in credited if n in by_n), default=0.0)
        quotes.append({"quote": quote[:120], "credited": credited, "best_source": best,
                       "best_overlap": round(overlap, 2), "credited_overlap": round(credited_overlap, 2),
                       "misattributed": bool(credited) and best not in credited and overlap - credited_overlap > 0.2})
    sentences = []
    for a, b in sentence_spans(text, 0, len(text)):
        sentence = text[a:b]
        cited = sorted({n for m in MARKER.finditer(sentence) for n in marker_numbers(m.group(1)) if n in by_n})
        if not cited:
            continue
        best, overlap = best_source(MARKER.sub("", sentence), answer.sources)
        credited_overlap = max(len(words(sentence) & words(by_n[n].text)) / max(1, len(words(sentence)))
                               for n in cited)
        sentences.append({"cited": cited, "best_source": best, "best_overlap": round(overlap, 2),
                          "credited_overlap": round(credited_overlap, 2),
                          "suspect": best not in cited and overlap - credited_overlap > 0.25})
    return {"quotes": quotes, "cited_sentences": sentences}


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--prompt-variant", default="v1", choices=sorted(PROMPTS))
    args.add_argument("--out", type=Path)
    parsed_args = args.parse_args()
    out = parsed_args.out or Path(__file__).with_name(f"answer_smoke_{parsed_args.prompt_variant}.jsonl")

    settings = get_settings()
    conn = db.open_db(settings.db_path)
    index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model, db.all_chunk_ids(conn))
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)
    answerer = Answerer(conn, DenseSearcher(conn, index, embedder), create_llm(settings),
                        HFTokenCounter(settings.llm_tokenizer), settings.ask_retrieve_k,
                        settings.ask_max_sources, settings.ask_context_tokens,
                        system_prompt=PROMPTS[parsed_args.prompt_variant])

    cases = [(q, doc) for q, doc, _, _ in QUESTIONS] + EXTRA + [(q, None) for q in UNANSWERABLE]
    records = []
    for question, expected_doc in cases:
        answer = answerer.ask(question)
        cited_docs = sorted({s.doc_id for s in answer.cited_sources})
        probe = attribution(answer)
        record = {
            "prompt_variant": parsed_args.prompt_variant,
            "question": question,
            "expected_doc": expected_doc,
            "answer": answer.text,
            "sources": [s.label for s in answer.sources],
            "cited": answer.citations.cited,
            "invalid": answer.citations.invalid,
            "sentences": answer.citations.n_sentences,
            "cited_sentences": answer.citations.n_cited_sentences,
            "cited_docs": cited_docs,
            "expected_doc_in_sources": expected_doc in {s.doc_id for s in answer.sources} if expected_doc else None,
            "expected_doc_cited": expected_doc in cited_docs if expected_doc else None,
            "misattributed_quotes": sum(q["misattributed"] for q in probe["quotes"]),
            "suspect_sentences": sum(s["suspect"] for s in probe["cited_sentences"]),
            "probe": probe,
            "prompt_tokens_counted": answer.prompt_tokens,
            "prompt_tokens_reported": answer.generation.prompt_tokens,
            "answer_tokens": answer.generation.completion_tokens,
            "generate_s": round(answer.timings["generate_s"], 2),
            "warnings": answer.warnings,
        }
        records.append(record)
        print(f"{'?' if expected_doc is None else ('✓' if record['expected_doc_cited'] else '✗')} "
              f"{record['generate_s']:5.1f}s cited {record['cited']} invalid {record['invalid']} "
              f"misquoted {record['misattributed_quotes']} suspect {record['suspect_sentences']} | {question[:60]}",
              flush=True)

    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    answerable = [r for r in records if r["expected_doc"]]
    print(f"\nexpected paper cited: {sum(r['expected_doc_cited'] for r in answerable)}/{len(answerable)} "
          f"(in sources: {sum(r['expected_doc_in_sources'] for r in answerable)}/{len(answerable)})")
    print(f"answers with any invalid citation: {sum(bool(r['invalid']) for r in records)}/{len(records)}")
    print(f"answers with no citation: {sum(not r['cited'] for r in records)}/{len(records)}")
    print(f"misattributed quotes: {sum(r['misattributed_quotes'] for r in records)}, "
          f"suspect cited sentences: {sum(r['suspect_sentences'] for r in records)}")
    print(f"generation seconds: {', '.join(str(r['generate_s']) for r in records)}")
    total_sentences = sum(r["sentences"] for r in records)
    print(f"cited sentences: {sum(r['cited_sentences'] for r in records)}/{total_sentences}")
    print(f"answers containing LaTeX: {sum(chr(92) + '(' in r['answer'] for r in records)}/{len(records)}")
    print(f"wrote {out}  (prompt {parsed_args.prompt_variant})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
