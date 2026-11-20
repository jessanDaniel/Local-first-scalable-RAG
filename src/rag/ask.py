"""Ask a question; get an answer with checked citations.

    rag ask "why do transformers divide attention scores by sqrt(d_k)?" [--explain]

--explain prints every stage: retrieved chunks, how they became numbered sources,
the exact prompt sent, model, token counts, timings, and each citation's check.
"""

import sys

from rag.answer.answerer import Answer, Answerer, ContextWindowError
from rag.chunk.tokens import HFTokenCounter
from rag.config import get_settings
from rag.embed.embedder import Embedder
from rag.index.dense import StaleIndexError, load_current
from rag.llm.factory import create_llm
from rag.llm.interface import LLMError
from rag.search import DenseSearcher
from rag.store import db

RULE = "=" * 78


def print_answer(answer: Answer, explain: bool) -> None:
    if explain:
        print(f"{RULE}\nRETRIEVAL ({len(answer.hits)} hits)\n{RULE}")
        used = {cid: s.n for s in answer.sources for cid in s.chunk_ids}
        for hit in answer.hits:
            where = f"-> [{used[hit.chunk.chunk_id]}]" if hit.chunk.chunk_id in used else "-> dropped"
            print(f"{hit.rank:>2}. {hit.score:.4f} {where:12} {hit.doc_title[:40]} §{hit.section_path[:50]}")
        print(f"\n{RULE}\nEXACT PROMPT ({answer.prompt_tokens} tokens, counted before sending)\n{RULE}")
        print(f"[system]\n{answer.system_prompt}\n\n[user]\n{answer.user_prompt}")
        g = answer.generation
        print(f"\n{RULE}\nGENERATION\n{RULE}")
        print(f"{g.provider} {g.model}  prompt tokens reported {g.prompt_tokens}, answer tokens {g.completion_tokens}")
        print("timings: " + ", ".join(f"{k} {v:.2f}" for k, v in answer.timings.items()))
        print(f"\n{RULE}\nANSWER\n{RULE}")
    print(answer.text.strip())
    print("\nSources cited:")
    for source in answer.cited_sources:
        print(f"  {source.label}\n      {source.link}")
    if not answer.cited_sources:
        print("  (none)")
    c = answer.citations
    print(f"\nCitation check: {len(c.cited)} valid, {len(c.invalid)} invalid, "
          f"{c.n_cited_sentences}/{c.n_sentences} sentences cited")
    for warning in answer.warnings:
        print(f"WARNING: {warning}")


def run(parsed_args) -> int:

    settings = get_settings()
    conn = db.open_db(settings.db_path)
    try:
        index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model,
                             db.all_chunk_ids(conn))
    except StaleIndexError as exc:
        print(exc)
        return 1
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction,
                        settings.embedding_batch_size)
    answerer = Answerer(
        conn,
        DenseSearcher(conn, index, embedder),
        create_llm(settings),
        HFTokenCounter(settings.llm_tokenizer),
        settings.ask_retrieve_k,
        settings.ask_max_sources,
        settings.ask_context_tokens,
    )
    try:
        answer = answerer.ask(parsed_args.question)
    except (ContextWindowError, LLMError) as exc:
        print(f"error: {exc}")
        return 1
    print_answer(answer, parsed_args.explain)
    return 0


def main() -> int:
    from rag.main import run_module

    return run_module(['ask'])


if __name__ == "__main__":
    sys.exit(main())
