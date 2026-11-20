"""Retrieval metrics over ranked chunk ids. Binary relevance: a chunk is gold or it is not.

hit@k     at least one gold chunk in the top k
recall@k  share of the question's gold chunks in the top k
MRR       1 / rank of the first gold chunk (0 if none was retrieved)
nDCG@k    rank-discounted gain, normalised by the best possible ordering
"""

import math


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    return next((i for i, chunk_id in enumerate(ranked, start=1) if chunk_id in gold), None)


def hit_at(ranked: list[str], gold: set[str], k: int) -> bool:
    return any(chunk_id in gold for chunk_id in ranked[:k])


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    rank = first_rank(ranked, gold)
    return 1 / rank if rank else 0.0


def ndcg_at(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, chunk_id in enumerate(ranked[:k], start=1) if chunk_id in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0
