"""question -> retrieve -> numbered sources -> LLM -> guardrails -> answer with checked citations."""

import time
from dataclasses import dataclass, field, replace

from rag.answer.citations import CitationReport, check_citations
from rag.answer.guardrails import Flag, check_answer_text, sanitize
from rag.answer.prompt import PROMPTS
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
    raw_text: str = ""  # what the model wrote; `text` is after guardrails (refusal citations removed)
    flags: list[Flag] = field(default_factory=list)
    refusal: bool = False
    prompt_version: str = ""
    ranks: dict[str, dict[str, int | None]] = field(default_factory=dict)  # chunk -> rank per retriever list
    retrieval_fallback: str | None = None  # why the reranker was skipped, if it was
    search_timings: dict[str, float] = field(default_factory=dict)

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.n in self.citations.cited]


@dataclass
class Prepared:
    """Retrieval and the prompt, ready for the model: the first half of an answer."""

    question: str
    hits: list[Hit]
    sources: list[Source]
    dropped_hits: list[Hit]
    user_prompt: str
    prompt_tokens: int
    removed_instructions: list[str]
    ranks: dict[str, dict[str, int | None]]
    retrieval_fallback: str | None
    search_timings: dict[str, float]
    started: float
    timings: dict[str, float]


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
        system_prompt: str | None = None,
        prompt_version: str = "v3",
        guardrails: bool = True,
        sanitize_sources: bool = False,
    ):
        default_system, self.format_prompt = PROMPTS[prompt_version]
        self.system_prompt = system_prompt or default_system
        self.prompt_version = prompt_version
        self.guardrails = guardrails
        self.sanitize_sources = sanitize_sources
        self._doc_texts: dict[str, str] = {}
        self.conn = conn
        self.searcher = searcher
        self.llm = llm
        self.prompt_counter = prompt_counter
        self.retrieve_k = retrieve_k
        self.max_sources = max_sources
        self.context_tokens = context_tokens

    def ask(self, question: str, collections: list[str] | None = None) -> Answer:
        """The whole pipeline in one call (the CLI). The API calls prepare, streams the
        generation itself, then calls finish: the same two halves, so the same prompt."""
        prepared = self.prepare(question, collections)
        return self.finish(prepared, self.llm.generate(prepared.user_prompt, system=self.system_prompt))

    def prepare(self, question: str, collections: list[str] | None = None) -> "Prepared":
        """Everything before the model: retrieve, assemble numbered sources, build and measure the
        prompt. Raises KeyError (unknown collection) or ContextWindowError."""
        start = time.perf_counter()
        hits = self.searcher.search(question, self.retrieve_k, collections)
        retrieved = time.perf_counter()
        # The searcher is shared between requests in the server: copy what it recorded now.
        ranks = dict(getattr(self.searcher, "last_ranks", None) or {})
        fallback = getattr(self.searcher, "last_fallback", None)
        search_timings = dict(getattr(self.searcher, "last_timings", None) or {})

        sections: dict[tuple[str, int], str] = {}

        def section_text(doc_id: str, section_ord: int) -> str:
            if (doc_id, section_ord) not in sections:
                for s in db.get_sections(self.conn, doc_id):
                    sections[(doc_id, s.ord)] = s.text
            return sections[(doc_id, section_ord)]

        sources, dropped = assemble_sources(
            hits, section_text, self.prompt_counter, self.context_tokens, self.max_sources
        )
        removed_instructions: list[str] = []
        prompt_sources = sources
        if self.sanitize_sources:  # optional: drop instruction-like sentences before the model reads them
            prompt_sources = []
            for source in sources:
                clean, removed = sanitize(source.text)
                removed_instructions += removed
                prompt_sources.append(replace(source, text=clean))
        user_prompt = self.format_prompt(question, prompt_sources)
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
        return Prepared(
            question=question,
            hits=hits,
            sources=sources,
            dropped_hits=dropped,
            user_prompt=user_prompt,
            prompt_tokens=prompt_tokens,
            removed_instructions=removed_instructions,
            ranks=ranks,
            retrieval_fallback=fallback,
            search_timings=search_timings,
            started=start,
            timings={"retrieve_s": retrieved - start, "assemble_s": assembled - retrieved},
        )

    def finish(self, prepared: "Prepared", generation: Generation) -> Answer:
        """Everything after the model: guardrails, citation check, warnings."""
        sources = prepared.sources
        text, flags, refusal = generation.text, [], False
        if self.guardrails:
            guard = check_answer_text(
                prepared.question,
                generation.text,
                {s.n: s.text for s in sources},
                {s.n: s.doc_id for s in sources},
                self.doc_texts({s.doc_id for s in sources}),
            )
            text, flags, refusal = guard.text, guard.flags, guard.refusal
            if prepared.removed_instructions:
                flags.append(
                    Flag(
                        "instructions_removed_from_sources",
                        f"{len(prepared.removed_instructions)} instruction-like sentence(s) removed before generation",
                    )
                )
        citations = check_citations(text, len(sources))

        warnings = []
        if citations.invalid:
            warnings.append(f"answer cites sources that do not exist: {citations.invalid}")
        if not citations.has_citations and not refusal:
            warnings.append("answer has no valid citations")
        if generation.completion_tokens and generation.completion_tokens >= self.llm.max_tokens:
            warnings.append(f"answer reached the {self.llm.max_tokens}-token limit and may be cut off")

        return Answer(
            question=prepared.question,
            text=text,
            sources=sources,
            citations=citations,
            hits=prepared.hits,
            dropped_hits=prepared.dropped_hits,
            generation=generation,
            system_prompt=self.system_prompt,
            user_prompt=prepared.user_prompt,
            prompt_tokens=prepared.prompt_tokens,
            timings={
                **prepared.timings,
                "generate_s": generation.elapsed_seconds,
                "total_s": time.perf_counter() - prepared.started,
            },
            warnings=warnings,
            raw_text=generation.text,
            flags=flags,
            refusal=refusal,
            prompt_version=self.prompt_version,
            ranks=prepared.ranks,
            retrieval_fallback=prepared.retrieval_fallback,
            search_timings=prepared.search_timings,
        )

    def doc_texts(self, doc_ids: set[str]) -> dict[str, str]:
        """Whole-document text (title + every section) per doc_id, cached: what the entity
        guardrail checks a named thing against."""
        for doc_id in doc_ids - self._doc_texts.keys():
            document = db.get_document(self.conn, doc_id)
            sections = " ".join(s.text for s in db.get_sections(self.conn, doc_id))
            self._doc_texts[doc_id] = f"{document.title if document else ''} {sections}"
        return {doc_id: self._doc_texts[doc_id] for doc_id in doc_ids}
