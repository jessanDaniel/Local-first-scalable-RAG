"""Phase 6 — Manual Top-K retrieval debug on a real document."""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.embeddings.embedder import Embedder
from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt
from rag.retrieval.similarity import print_retrieval_debug, top_k


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)
    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)

    query = "What is the company's refund policy?"
    query_vec = embedder.embed_query(query)
    results = top_k(query_vec, embedded, k=3)

    print_retrieval_debug(query, results)


if __name__ == "__main__":
    main()