"""The exact text sent to the LLM: a fixed system prompt plus sources and question.

The rules target failures seen in the learning track: inventing facts when the
context is thin (Phase 12), and a bare "I don't know" when the answer was there
but split across chunks (Phase 14) — so a partial answer must say what is missing
instead of refusing outright. This is prompt v2: compared with v1 on 13 questions
in experiments/stage1/answer_seed10.md (fewer uncited sentences, no misattributed
quotes, little LaTeX; but refusals still sometimes carry citations).
"""

from rag.answer.sources import Source, format_source

SYSTEM_PROMPT = """You answer questions using only the numbered sources you are given.

Rules:
1. Every sentence that states a fact must end with the number of the source it comes from, in square brackets, like [2] or [1][3]. If you cannot point to a source for a sentence, leave the sentence out.
2. Cite only the source numbers given to you, and only a source whose text says what the sentence says.
3. Do not use outside knowledge, even if you know the answer.
4. Use quotation marks only for words copied exactly from the source you cite.
5. If the sources answer only part of the question, answer that part, then say plainly which part they do not cover. A sentence saying something is missing gets no citation.
6. If the sources do not answer the question at all, say so in one sentence, naming what is missing, with no citation.
7. Write math in plain text, for example sqrt(d_k), not LaTeX.
8. Be concise: a short paragraph unless the question asks for more."""


def build_user_prompt(question: str, sources: list[Source]) -> str:
    blocks = "\n\n".join(format_source(source) for source in sources)
    return f"Sources:\n\n{blocks}\n\nQuestion: {question}"
