"""Retrieved hits -> the numbered sources an answer may cite.

- Hits from the same section whose spans overlap or touch (step 6's one-sentence
  overlap, or two neighbouring chunks) are merged into one source, so the model
  never reads the same sentence twice under two numbers.
- Sources keep retrieval order (best rank first) and are numbered 1..n.
- Sources are added while the sources block stays within budget_tokens (counted
  with the LLM's tokenizer) and there are fewer than max_sources; a source that
  does not fit is skipped and the next, smaller one is tried.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from rag.chunk.tokens import TokenCounter
from rag.search import Hit


@dataclass
class Source:
    n: int
    doc_id: str
    title: str
    section_path: str
    page_start: int | None
    page_end: int | None
    text: str
    chunk_ids: list[str]
    best_rank: int
    best_score: float
    source_uri: str | None
    n_tokens: int = 0  # of its block in the prompt

    @property
    def pages(self) -> str:
        if self.page_start is None:
            return "p.?"
        if self.page_end in (None, self.page_start):
            return f"p.{self.page_start}"
        return f"pp.{self.page_start}-{self.page_end}"

    @property
    def label(self) -> str:
        """How a reader sees the source: '[2] Attention Is All You Need (arxiv:1706.03762) §3.2.1 ..., p.4'."""
        return f"[{self.n}] {self.title} ({self.doc_id}) §{self.section_path}, {self.pages}"

    @property
    def link(self) -> str | None:
        """Where to open it, at the right page when known."""
        page = f"#page={self.page_start}" if self.page_start else ""
        if self.doc_id.startswith("arxiv:"):
            return f"https://arxiv.org/pdf/{self.doc_id.split(':', 1)[1]}{page}"
        if self.source_uri and self.source_uri.startswith("/"):
            return f"file://{self.source_uri}{page}"
        return self.source_uri


@dataclass
class _Group:
    doc_id: str
    section_ord: int
    start: int
    end: int
    hits: list[Hit] = field(default_factory=list)


def merge_hits(hits: list[Hit], section_text: Callable[[str, int], str]) -> list[_Group]:
    """Merge hits from the same section whose spans overlap or are separated only by whitespace."""
    by_section: dict[tuple[str, int], list[Hit]] = {}
    for hit in hits:
        by_section.setdefault((hit.chunk.doc_id, hit.chunk.section_ord), []).append(hit)
    groups: list[_Group] = []
    for (doc_id, section_ord), section_hits in by_section.items():
        text = section_text(doc_id, section_ord)
        section_hits.sort(key=lambda h: h.chunk.char_start)
        current: _Group | None = None
        for hit in section_hits:
            c = hit.chunk
            touches = current is not None and (
                c.char_start <= current.end or not text[current.end : c.char_start].strip()
            )
            if touches:
                current.end = max(current.end, c.char_end)
                current.hits.append(hit)
            else:
                current = _Group(doc_id, section_ord, c.char_start, c.char_end, [hit])
                groups.append(current)
    groups.sort(key=lambda g: min(h.rank for h in g.hits))
    return groups


def format_source(source: Source) -> str:
    """The block the LLM reads for one source."""
    return f"[{source.n}] {source.title} — {source.section_path} ({source.pages})\n{source.text}"


def assemble_sources(
    hits: list[Hit],
    section_text: Callable[[str, int], str],
    counter: TokenCounter,
    budget_tokens: int,
    max_sources: int,
) -> tuple[list[Source], list[Hit]]:
    """Numbered sources within the budget, and the hits that did not make it in."""
    sources: list[Source] = []
    dropped: list[Hit] = []
    used = 0
    for group in merge_hits(hits, section_text):
        first = min(group.hits, key=lambda h: h.rank)
        pages_start = [h.chunk.page_start for h in group.hits if h.chunk.page_start is not None]
        pages_end = [h.chunk.page_end for h in group.hits if h.chunk.page_end is not None]
        source = Source(
            n=len(sources) + 1,
            doc_id=group.doc_id,
            title=first.doc_title,
            section_path=first.section_path,
            page_start=min(pages_start) if pages_start else None,
            page_end=max(pages_end) if pages_end else None,
            text=section_text(group.doc_id, group.section_ord)[group.start : group.end],
            chunk_ids=[
                h.chunk.chunk_id for h in sorted(group.hits, key=lambda h: (h.chunk.char_start, h.chunk.char_end))
            ],
            best_rank=first.rank,
            best_score=first.score,
            source_uri=first.source_uri,
        )
        source.n_tokens = counter.count(format_source(source)) + 2  # + the blank line between blocks
        if len(sources) < max_sources and used + source.n_tokens <= budget_tokens:
            sources.append(source)
            used += source.n_tokens
        else:
            dropped.extend(group.hits)
    return sources, dropped
