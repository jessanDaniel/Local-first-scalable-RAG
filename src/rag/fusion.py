"""Reciprocal Rank Fusion: merge ranked lists using positions only, never scores.

BM25 scores run 0-30, cosine similarities 0-1; their sizes mean different
things, so adding them would need a weighting we cannot justify. RRF ignores
scores. Each list gives a chunk 1 / (k + its rank there); a chunk missing from
a list gets nothing from it:

    fused(chunk) = sum over lists of 1 / (k + rank)

k = 60 (Cormack, Clarke & Buettcher, SIGIR 2009) keeps rank 1 from dominating:
1/61 vs 1/62 for ranks 1 and 2, so agreement between the lists matters more
than a single first place.
"""

from dataclasses import dataclass

RRF_K = 60


@dataclass
class Fused:
    chunk_id: str
    score: float
    ranks: dict[str, int | None]  # list name -> rank in that list (None: not in it)


def rrf(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> list[Fused]:
    """Fuse named lists of chunk ids (best first). Ties go to the better best-rank, then the id,
    so the order is deterministic."""
    positions = {
        name: {chunk_id: rank for rank, chunk_id in enumerate(ids, start=1)} for name, ids in ranked_lists.items()
    }
    all_ids = dict.fromkeys(chunk_id for ids in ranked_lists.values() for chunk_id in ids)
    fused = []
    for chunk_id in all_ids:
        ranks = {name: positions[name].get(chunk_id) for name in ranked_lists}
        score = sum(1 / (k + rank) for rank in ranks.values() if rank is not None)
        fused.append(Fused(chunk_id, score, ranks))
    fused.sort(key=lambda f: (-f.score, min(r for r in f.ranks.values() if r is not None), f.chunk_id))
    return fused
