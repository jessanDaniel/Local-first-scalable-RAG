"""Phase 8 — exercise Retriever.index_chunks() + retrieve() end-to-end.

Same doc/chunking/query as Phase 6 and 7, so the output is directly
comparable to both. Reuses print_retrieval_debug from Phase 6 — same
ScoredChunk contract, regardless of backend."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.retrieval.retriever import Retriever
from src.retrieval.similarity import print_retrieval_debug
from src.retrieval.vector_store import VectorStore


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)

    embedder = Embedder()
    vector_store = VectorStore(
        persist_path=project_root / "chroma_db",
        collection_name="company_policy_retriever",
    )
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    retriever.index_chunks(chunks)

    query = "What is the company's refund policy?"
    results = retriever.retrieve(query, k=3)

    print_retrieval_debug(query, results)


if __name__ == "__main__":
    main()