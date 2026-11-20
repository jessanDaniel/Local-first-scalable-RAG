"""BM25 keyword search, written by hand (the "manual before magic" step before any library).

Three pieces:

1. Inverted index: for every term, the chunks that contain it and how often.

       "34.81" -> [(chunk 17, 1)]
       "bleu"  -> [(chunk 17, 2), (chunk 18, 1), (chunk 40, 3), ...]

   A query only touches the chunks that share a term with it, never the rest.

2. IDF (inverse document frequency): how rare a term is across all N chunks.

       idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))     df = chunks containing t

   "34.81" in 1 of 400 chunks -> idf ~ 5.5;  "model" in 300 of 400 -> idf ~ 0.3.
   (This is the Lucene form of BM25's IDF: the "1 +" keeps it positive even for
   a term in more than half the chunks.)

3. The score of one chunk for a query: the sum, over the query's terms, of

       idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

   tf       how often the term occurs in this chunk
   k1=1.2   saturation: the 2nd occurrence adds less than the 1st, the 10th
            almost nothing (as k1 -> 0, only "present or not" counts)
   b=0.75   length normalisation: a chunk longer than average needs more
            occurrences for the same score (b=0: length ignored)

Everything is in memory and rebuilt from SQLite in milliseconds at this size,
so there is no index file that could go stale.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from rag.index.tokenize import query_terms, tokenize

K1 = 1.2
B = 0.75


@dataclass
class TermScore:
    """One query term's contribution to one chunk's score (for --explain)."""

    term: str
    tf: int
    df: int
    idf: float
    score: float


class BM25Index:
    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.lengths: list[int] = []  # tokens per chunk
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(row, tf)]
        self.avg_length = 0.0

    @classmethod
    def build(cls, documents: list[tuple[str, str]], k1: float = K1, b: float = B) -> "BM25Index":
        """documents: (chunk_id, text to index)."""
        index = cls(k1, b)
        for row, (chunk_id, text) in enumerate(documents):
            tokens = tokenize(text)
            index.chunk_ids.append(chunk_id)
            index.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                index.postings[term].append((row, tf))
        index.avg_length = sum(index.lengths) / len(index.lengths) if index.lengths else 0.0
        return index

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log(1 + (self.n_chunks - df + 0.5) / (df + 0.5))

    def term_score(self, tf: int, length: int, idf: float) -> float:
        normalised_length = 1 - self.b + self.b * length / self.avg_length
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * normalised_length)

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score), best first. `allowed` restricts scoring to those chunks
        (the collection filter), so a small collection still gets its full top-k."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms(query):
            postings = self.postings.get(term)
            if not postings:
                continue  # a term no chunk contains adds nothing
            idf = self.idf(term)
            for row, tf in postings:
                if allowed is not None and self.chunk_ids[row] not in allowed:
                    continue
                scores[row] += self.term_score(tf, self.lengths[row], idf)
        # Ties broken by row (insertion order), so results are deterministic.
        best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.chunk_ids[row], score) for row, score in best]

    def explain(self, query: str, chunk_id: str) -> list[TermScore]:
        """How each query term contributed to this chunk's score (terms it lacks score 0)."""
        row = self.chunk_ids.index(chunk_id)
        parts = []
        for term in query_terms(query):
            tf = next((count for r, count in self.postings.get(term, ()) if r == row), 0)
            idf = self.idf(term) if self.df(term) else 0.0
            score = self.term_score(tf, self.lengths[row], idf) if tf else 0.0
            parts.append(TermScore(term, tf, self.df(term), idf, score))
        return parts
