"""Phase 8 — retrieve(query, k): decoupled from prompts and LLM providers.
Knows about Embedder, VectorStore, and Chunk/ScoredChunk. Knows nothing about
prompt templates, context construction, or any LLM provider."""


from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunk
from src.retrieval.similarity import ScoredChunk
from src.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def index_chunks(self, chunks: list[Chunk]) -> None:
        embedded = self.embedder.embed_chunks(chunks)
        self.vector_store.add(
            ids=[ec.chunk.chunk_id for ec in embedded],
            embeddings=[ec.embedding.tolist() for ec in embedded],
            documents=[ec.chunk.text for ec in embedded],
            metadata=[
                {"document_id": ec.chunk.document_id, "position": ec.chunk.position}
                for ec in embedded
            ],
        )
    
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.query(query_vec.tolist(), k)

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        scored: list[ScoredChunk] = []

        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(ids, docs, metas, distances), start=1
        ):
            scored.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    position=meta["position"],
                    text=text,
                    score=1 - distance,
                    rank=rank,
                )
            )
        return scored