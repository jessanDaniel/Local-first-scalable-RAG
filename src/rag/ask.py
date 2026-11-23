"""Ask a question; get an answer with checked citations.

    rag ask "why do transformers divide attention scores by sqrt(d_k)?" [--mode hybrid|dense|bm25]
            [--rerank off|fused|replace] [--explain]

Retrieval uses the RETRIEVAL_MODE setting (hybrid by default) unless --mode says otherwise, and
the ASK_RERANK setting (fused by default) unless --rerank says otherwise; --rerank off skips the
reranker entirely (its model is not even loaded).

--explain prints every stage: retrieved chunks, how they became numbered sources,
the exact prompt sent, model, token counts, timings, and each citation's check.
"""

import sys

from rag.answer.answerer import Answer, ContextWindowError
from rag.config import get_settings
from rag.index.current import StaleIndexError
from rag.llm.interface import LLMError
from rag.pipeline import Pipeline
from rag.search import describe_ranks

RULE = "=" * 78


def print_answer(answer: Answer, explain: bool, mode: str = "dense", ranks: dict | None = None) -> None:
    if explain:
        print(f"{RULE}\nRETRIEVAL ({mode}, {len(answer.hits)} hits)\n{RULE}")
        used = {cid: s.n for s in answer.sources for cid in s.chunk_ids}
        for hit in answer.hits:
            where = f"-> [{used[hit.chunk.chunk_id]}]" if hit.chunk.chunk_id in used else "-> dropped"
            origin = f" ({describe_ranks(ranks[hit.chunk.chunk_id])})" if ranks and hit.chunk.chunk_id in ranks else ""
            print(f"{hit.rank:>2}. {hit.score:.4f} {where:12} {hit.doc_title[:40]} §{hit.section_path[:50]}{origin}")
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
    print(
        f"\nCitation check: {len(c.cited)} valid, {len(c.invalid)} invalid, "
        f"{c.n_cited_sentences}/{c.n_sentences} sentences cited"
    )
    for warning in answer.warnings:
        print(f"WARNING: {warning}")
    for flag in answer.flags:
        print(f"FLAG {flag.kind}: {flag.detail}" + (f"\n      in: {flag.sentence[:160]}" if flag.sentence else ""))
    if explain and answer.raw_text.strip() != answer.text.strip():
        print(f"\n(guardrails edited the answer; the model wrote:)\n{answer.raw_text.strip()}")


def run(parsed_args) -> int:

    settings = get_settings()
    pipeline = Pipeline(settings)
    mode = parsed_args.mode or settings.retrieval_mode
    try:
        answerer = pipeline.answerer(mode, parsed_args.rerank or settings.ask_rerank)
    except StaleIndexError as exc:
        print(exc)
        return 1
    searcher = answerer.searcher
    try:
        answer = answerer.ask(parsed_args.question, parsed_args.collections)
    except (ContextWindowError, LLMError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyError as exc:  # unknown collection
        print(f"error: {exc.args[0]}")
        return 1
    label = mode if searcher.mode != "rerank" else f"{mode} + rerank {searcher.strategy}"
    print_answer(answer, parsed_args.explain, label, answer.ranks)
    if answer.retrieval_fallback:
        print(f"WARNING: {answer.retrieval_fallback}")
    return 0


def main() -> int:
    from rag.main import run_module

    return run_module(["ask"])


if __name__ == "__main__":
    sys.exit(main())
