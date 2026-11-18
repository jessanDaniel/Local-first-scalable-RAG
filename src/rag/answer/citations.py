"""Check the [n] citations in an answer against the sources the model was given.

This proves a citation points at a real source. It does not prove the source
supports the sentence (groundedness) — that belongs to the evaluation stage.
"""

import re
from dataclasses import dataclass

from rag.chunk.chunker import sentence_spans

MARKER = re.compile(r"\[(\s*\d+\s*(?:[,;–-]\s*\d+\s*)*)\]")


@dataclass
class CitationReport:
    cited: list[int]  # valid source numbers the answer uses, ascending
    invalid: list[int]  # numbers that match no source
    n_sentences: int
    n_cited_sentences: int

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)


def marker_numbers(marker_body: str) -> list[int]:
    """'2' -> [2]; '1, 3' -> [1, 3]; '2-4' -> [2, 3, 4]."""
    numbers: list[int] = []
    for part in re.split(r"[,;]", marker_body):
        bounds = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
        if len(bounds) == 2 and bounds[0] <= bounds[1] and bounds[1] - bounds[0] < 50:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            numbers.extend(bounds)
    return numbers


def check_citations(answer: str, n_sources: int) -> CitationReport:
    found: set[int] = set()
    for match in MARKER.finditer(answer):
        found.update(marker_numbers(match.group(1)))
    sentences = [answer[a:b] for a, b in sentence_spans(answer, 0, len(answer)) if answer[a:b].strip()]
    return CitationReport(
        cited=sorted(n for n in found if 1 <= n <= n_sources),
        invalid=sorted(n for n in found if not 1 <= n <= n_sources),
        n_sentences=len(sentences),
        n_cited_sentences=sum(1 for s in sentences if MARKER.search(s)),
    )
