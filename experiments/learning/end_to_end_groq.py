"""Phase 11 — same full pipeline as end_to_end_debug.py, but with GroqLLM
swapped in for OllamaLLM. Retrieval, chunking, and context construction code
is completely untouched — only the LLM changed."""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.embeddings.embedder import Embedder
from rag.generation.context_builder import build_prompt, print_context_debug
from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt
from rag.llm.groq import GroqLLM
from rag.llm.ollama import print_generation_debug
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)

    embedder = Embedder()
    vector_store = VectorStore(
        persist_path=project_root / "chroma_db",
        collection_name="company_policy_end_to_end",
    )
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    retriever.index_chunks(chunks)

    query = "What is the company's refund policy?"
    results = retriever.retrieve(query, k=3)

    print_context_debug(query, results)

    prompt = build_prompt(query, results)
    llm = GroqLLM()
    result = llm.generate(prompt)

    print()
    print_generation_debug(result)


if __name__ == "__main__":
    main()