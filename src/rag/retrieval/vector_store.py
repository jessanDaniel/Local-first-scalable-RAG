"""Phase 8 — thin wrapper around a Chroma collection.
This is the storage/index layer only: ids, vectors, documents, metadata in;
raw Chroma results out. No knowledge of embedding models, chunks, prompts,
or LLM providers lives here."""

from pathlib import Path
import chromadb

class VectorStore:
    def __init__(self, persist_path: Path, collection_name: str):
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space":"cosine"}
            )
        
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadata: list[dict],
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
        )

    def query(self, query_embedding: list[float], k: int) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )