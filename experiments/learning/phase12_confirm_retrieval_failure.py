"""Phase 12 — confirm the retrieval-failure case at different k values."""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.embeddings.embedder import Embedder
from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore


def main() -> None:
    doc = load_txt(project_root / "documents" / "employee_handbook.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=150, overlap=15)

    embedder = Embedder()
    vector_store = VectorStore(
        persist_path=project_root / "chroma_db",
        collection_name="employee_handbook_phase12",
    )
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    # already indexed from the exploration script; index_chunks again is safe
    # to skip here since ids already exist — just query.

    query = "How many consecutive late arrivals trigger a review?"
    needle = "three late arrivals"

    for k in (1, 2, 3, 5):
        results = retriever.retrieve(query, k=k)
        found = any(needle.lower() in r.text.lower() for r in results)
        print(f"k={k}: correct chunk retrieved = {found}")
        for r in results:
            preview = r.text[:70].replace("\n", " ")
            print(f"    rank={r.rank} score={r.score:.4f} chunk={r.chunk_id} \"{preview}...\"")


if __name__ == "__main__":
    main()
