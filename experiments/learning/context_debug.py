"""Phase 9 — end-to-end: retrieve(query, k) -> build the exact prompt string
that would be sent to an LLM. Still no LLM call anywhere in this script."""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.embeddings.embedder import Embedder
from rag.generation.context_builder import print_context_debug
from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)

    embedder = Embedder()
    vector_store = VectorStore(
        persist_path=project_root / "chroma_db",
        collection_name="company_policy_context_debug",
    )
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    retriever.index_chunks(chunks)

    query = "What is the company's refund policy?"
    results = retriever.retrieve(query, k=3)

    print_context_debug(query, results)


if __name__ == "__main__":
    main()