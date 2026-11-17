"""Phase 7 step 2 — Recreate the collection with cosine distance space and confirm
scores now numerically match Phase 6's manual cosine similarities.

For a collection with metadata={"hnsw:space": "cosine"}, Chroma reports
distance = 1 - cosine_similarity, so similarity = 1 - distance."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import chromadb

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)

    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)

    client = chromadb.PersistentClient(path=str(project_root / "chroma_db"))
    collection = client.get_or_create_collection(
        name="company_policy_cosine",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[ec.chunk.chunk_id for ec in embedded],
        embeddings=[ec.embedding.tolist() for ec in embedded],
        documents=[ec.chunk.text for ec in embedded],
        metadatas=[
            {"document_id": ec.chunk.document_id, "position": ec.chunk.position}
            for ec in embedded
        ],
    )

    query = "What is the company's refund policy?"
    query_vec = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Question:\n  {query}\n")
    print("Retrieved chunks (Chroma, cosine distance space):\n")
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    for rank, (chunk_id, text, meta, distance) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        similarity = 1 - distance
        preview = text if len(text) <= 240 else text[:240] + "..."
        print(f"Rank: {rank}")
        print(f"Distance: {distance:.4f}  ->  Similarity (1 - distance): {similarity:.4f}")
        print(f"Source: {meta['document_id']}")
        print(f"Chunk: {chunk_id} (position={meta['position']})")
        print(f'"{preview}"')
        print()


if __name__ == "__main__":
    main()
