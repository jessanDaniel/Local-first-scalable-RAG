"""Phase 7 — Same retrieval task as Phase 6, but backed by Chroma instead of
manual NumPy top-K. Uses the *default* Chroma collection settings on purpose —
we'll compare the numbers to Phase 6 and see they don't match, then figure out why."""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import chromadb

from rag.embeddings.embedder import Embedder
from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)

    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)

    # Persistent on disk — same collection survives across script runs.
    client = chromadb.PersistentClient(path=str(project_root / "chroma_db"))
    collection = client.get_or_create_collection(name="company_policy_default")

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
    print("Retrieved chunks (Chroma, default distance metric):\n")
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    for rank, (chunk_id, text, meta, distance) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        preview = text if len(text) <= 240 else text[:240] + "..."
        print(f"Rank: {rank}")
        print(f"Distance: {distance:.4f}")
        print(f"Source: {meta['document_id']}")
        print(f"Chunk: {chunk_id} (position={meta['position']})")
        print(f'"{preview}"')
        print()


if __name__ == "__main__":
    main()