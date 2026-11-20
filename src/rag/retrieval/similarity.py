"""Phase 6 — Manual cosine similarity + Top-K (no vector DB yet)."""

from dataclasses import dataclass

import numpy as np

from rag.embeddings.embedder import EmbeddedChunk


@dataclass
class ScoredChunk:
    chunk_id: str
    document_id: str
    position: int
    text: str
    score: float
    rank: int


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def top_k(
    query_embedding: np.ndarray,
    embedded_chunks: list[EmbeddedChunk],
    k: int = 3,
) -> list[ScoredChunk]:
    """Score every chunk against the query; return the top K by cosine similarity."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not embedded_chunks:
        return []

    # Stack chunk vectors into a matrix: shape (n_chunks, dims)
    matrix = np.vstack([ec.embedding for ec in embedded_chunks])
    # Normalize for cosine: sim = (matrix @ q) / (|rows| * |q|)
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(matrix, axis=1)
    scores = (matrix @ query_embedding) / (chunk_norms * query_norm)

    # Argsort ascending → reverse for highest scores first
    order = np.argsort(scores)[::-1]
    top_indices = order[: min(k, len(embedded_chunks))]

    results: list[ScoredChunk] = []
    for rank, idx in enumerate(top_indices, start=1):
        ec = embedded_chunks[int(idx)]
        results.append(
            ScoredChunk(
                chunk_id=ec.chunk.chunk_id,
                document_id=ec.chunk.document_id,
                position=ec.chunk.position,
                text=ec.chunk.text,
                score=float(scores[int(idx)]),
                rank=rank,
            )
        )
    return results


def print_retrieval_debug(query: str, results: list[ScoredChunk]) -> None:
    """Retrieval debug mode — what the retriever actually returned."""
    print(f"Question:\n  {query}\n")
    print("Retrieved chunks:\n")
    for r in results:
        print(f"Rank: {r.rank}")
        print(f"Score: {r.score:.4f}")
        print(f"Source: {r.document_id}")
        print(f"Chunk: {r.chunk_id} (position={r.position})")
        preview = r.text if len(r.text) <= 240 else r.text[:240] + "..."
        print(f'"{preview}"')
        print()
