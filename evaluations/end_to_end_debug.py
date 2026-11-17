"""Phase 10 — the full pipeline, first time end-to-end:
retrieve(query, k) -> build_prompt(query, chunks) -> Ollama -> answer.

Every stage prints its own debug output — nothing hidden."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedder import Embedder
from src.generation.context_builder import build_prompt, print_context_debug
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.llm.ollama import OllamaLLM, print_generation_debug
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


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
    llm = OllamaLLM(model="llama3.2:1b")
    result = llm.generate(prompt)

    print()
    print_generation_debug(result)


if __name__ == "__main__":
    main()