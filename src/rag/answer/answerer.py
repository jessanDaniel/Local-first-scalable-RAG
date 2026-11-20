"""question -> retrieve -> numbered sources -> LLM -> answer with checked citations."""

import time
from dataclasses import dataclass, field

from rag.answer.citations import CitationReport, check_citations
from rag.answer.prompt import SYSTEM_PROMPT, build_user_prompt
from rag.answer.sources import Source, assemble_sources
from rag.chunk.tokens import TokenCounter
from rag.llm.interface import LLM, Generation
from rag.search import Hit
from rag.store import db

# Tokens the chat template adds around the system and user messages (role markers etc.).
CHAT_TEMPLATE_OVERHEAD = 32


class ContextWindowError(RuntimeError):
    """The prompt plus the answer would not fit the model's context window."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source]
    citations: CitationReport
    hits: list[Hit]
    dropped_hits: list[Hit]
    generation: Generation
    system_prompt: str
    user_prompt: str
    prompt_tokens: int  # counted before sending, with the LLM's tokenizer
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.n in self.citations.cited]


class Answerer:
    def __init__(
        self,
        conn,
        searcher,
        llm: LLM,
        prompt_counter: TokenCounter,
        retrieve_k: int,
        max_sources: int,
        context_tokens: int,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.system_prompt = system_prompt
        self.conn = conn
        self.searcher = searcher
        self.llm = llm
        self.prompt_counter = prompt_counter
        self.retrieve_k = retrieve_k
        self.max_sources = max_sources
        self.context_tokens = context_tokens

    def ask(self, question: str, collections: list[str] | None = None) -> Answer:
        start = time.perf_counter()
        hits = self.searcher.search(question, self.retrieve_k, collections)
        retrieved = time.perf_counter()

        sections: dict[tuple[str, int], str] = {}

        def section_text(doc_id: str, section_ord: int) -> str:
            if (doc_id, section_ord) not in sections:
                for s in db.get_sections(self.conn, doc_id):
                    sections[(doc_id, s.ord)] = s.text
            return sections[(doc_id, section_ord)]

        sources, dropped = assemble_sources(
            hits, section_text, self.prompt_counter, self.context_tokens, self.max_sources
        )
        user_prompt = build_user_prompt(question, sources)
        prompt_tokens = (
            self.prompt_counter.count(self.system_prompt)
            + self.prompt_counter.count(user_prompt)
            + CHAT_TEMPLATE_OVERHEAD
        )
        if prompt_tokens + self.llm.max_tokens > self.llm.context_window:
            raise ContextWindowError(
                f"prompt {prompt_tokens} + answer {self.llm.max_tokens} tokens > context window "
                f"{self.llm.context_window} of {self.llm.model}; the model would silently lose the start"
            )
        assembled = time.perf_counter()

        generation = self.llm.generate(user_prompt, system=self.system_prompt)
        citations = check_citations(generation.text, len(sources))

        warnings = []
        if citations.invalid:
            warnings.append(f"answer cites sources that do not exist: {citations.invalid}")
        if not citations.has_citations:
            warnings.append("answer has no valid citations")
        if generation.completion_tokens and generation.completion_tokens >= self.llm.max_tokens:
            warnings.append(f"answer reached the {self.llm.max_tokens}-token limit and may be cut off")

        return Answer(
            question=question,
            text=generation.text,
            sources=sources,
            citations=citations,
            hits=hits,
            dropped_hits=dropped,
            generation=generation,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            prompt_tokens=prompt_tokens,
            timings={
                "retrieve_s": retrieved - start,
                "assemble_s": assembled - retrieved,
                "generate_s": generation.elapsed_seconds,
                "total_s": time.perf_counter() - start,
            },
            warnings=warnings,
        )
