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

Statistics over the chunks being searched (step 24). With a collection filter, N, the average
length and every df are counted over the searched chunks only, not the whole corpus. So a
collection's ranking depends only on that collection: adding someone else's folder of documents
(which makes "leave" or "password" common corpus-wide) cannot move it. Without a filter, the
statistics are the whole corpus's, exactly as before.
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
        self.row_of: dict[str, int] = {}

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
        index.row_of = {chunk_id: row for row, chunk_id in enumerate(index.chunk_ids)}
        return index

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        return idf_of(self.df(term), self.n_chunks)

    def term_score(self, tf: int, length: int, idf: float, avg_length: float | None = None) -> float:
        normalised_length = 1 - self.b + self.b * length / (avg_length or self.avg_length)
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * normalised_length)

    def scope(self, allowed: set[str] | None) -> tuple[set[int] | None, int, float]:
        """(rows searched or None for all, N, average length) for a collection filter."""
        if allowed is None:
            return None, self.n_chunks, self.avg_length
        rows = {self.row_of[c] for c in allowed if c in self.row_of}
        return rows, len(rows), (sum(self.lengths[r] for r in rows) / len(rows) if rows else 0.0)

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score), best first. `allowed` restricts scoring to those chunks (the
        collection filter) and the statistics to them too, so a small collection still gets its full
        top-k and its ranking doesn't depend on other collections."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        rows, n, avg_length = self.scope(allowed)
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms(query):
            postings = self.postings.get(term)
            if rows is not None and postings:
                postings = [(row, tf) for row, tf in postings if row in rows]
            if not postings:
                continue  # a term no searched chunk contains adds nothing
            idf = idf_of(len(postings), n)
            for row, tf in postings:
                scores[row] += self.term_score(tf, self.lengths[row], idf, avg_length)
        # Ties broken by row (insertion order), so results are deterministic.
        best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.chunk_ids[row], score) for row, score in best]

    def explain(self, query: str, chunk_id: str, allowed: set[str] | None = None) -> list[TermScore]:
        """How each query term contributed to this chunk's score (terms it lacks score 0), with the
        same statistics the search used."""
        row = self.row_of[chunk_id]
        rows, n, avg_length = self.scope(allowed)
        parts = []
        for term in query_terms(query):
            postings = [(r, c) for r, c in self.postings.get(term, ()) if rows is None or r in rows]
            tf = next((count for r, count in postings if r == row), 0)
            idf = idf_of(len(postings), n) if postings else 0.0
            score = self.term_score(tf, self.lengths[row], idf, avg_length) if tf else 0.0
            parts.append(TermScore(term, tf, len(postings), idf, score))
        return parts


def idf_of(df: int, n: int) -> float:
    return math.log(1 + (n - df + 0.5) / (df + 0.5))
