"""Phase 9 — deterministic context construction: combine the question and
retrieved chunks into the exact text that would be sent to an LLM.
No LLM call happens here — this only builds and inspects the prompt string."""

from rag.retrieval.similarity import ScoredChunk

PROMPT_TEMPLATE = """Answer the question using only the context below. If the context does not contain the answer, say you don't know.
Context:
{context}
Question: {question}
Answer:"""


def build_context(chunks: list[ScoredChunk]) -> str:
    parts = [f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks]
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    context = build_context(chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def print_context_debug(question: str, chunks: list[ScoredChunk]) -> None:
    prompt = build_prompt(question, chunks)
    print("=" * 60)
    print("EXACT TEXT SENT TO LLM:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"Prompt length: {len(prompt)} characters, {len(prompt.split())} words")
