"""Phase 12 — mandatory grounding + failure experiments.

Three experiments, each explicitly diagnosing retrieval vs. generation:
1. Missing-information question -> correctly refused vs. hallucinated.
2. Retrieval-failure case -> answer exists, but is the right chunk retrieved?
3. Generation-failure case -> correct chunk retrieved, does the LLM use it right?
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedder import Embedder
from src.generation.context_builder import build_prompt, print_context_debug
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.llm.groq import GroqLLM
from src.llm.ollama import print_generation_debug
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


_indexed_collections: set[str] = set()


def run(label: str, doc_name: str, collection: str, chunk_size: int, overlap: int, query: str, k: int) -> None:
    print("\n" + "#" * 70)
    print(f"# {label}")
    print("#" * 70)

    doc = load_txt(project_root / "documents" / doc_name)
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=chunk_size, overlap=overlap)

    embedder = Embedder()
    vector_store = VectorStore(persist_path=project_root / "chroma_db", collection_name=collection)
    retriever = Retriever(vector_store=vector_store, embedder=embedder)

    # Index only if the collection is actually empty — avoids the empty-context
    # bug from the first attempt, and avoids duplicate-id errors against
    # collections already populated by earlier exploration scripts.
    if collection not in _indexed_collections and vector_store.collection.count() == 0:
        retriever.index_chunks(chunks)
    _indexed_collections.add(collection)

    results = retriever.retrieve(query, k=k)
    print_context_debug(query, results)

    prompt = build_prompt(query, results)
    llm = GroqLLM()
    result = llm.generate(prompt)
    print()
    print_generation_debug(result)


def main() -> None:
    # Experiment 1: missing information
    run(
        label="Experiment 1 — Missing information",
        doc_name="company_policy.txt",
        collection="company_policy_phase12_missing",
        chunk_size=100,
        overlap=10,
        query="Does Acme Corp offer any discounts for students or military members?",
        k=3,
    )

    # Experiment 2: retrieval failure (k=1, deliberately too narrow)
    run(
        label="Experiment 2a — Retrieval failure (k=1)",
        doc_name="employee_handbook.txt",
        collection="employee_handbook_phase12",
        chunk_size=150,
        overlap=15,
        query="How many consecutive late arrivals trigger a review?",
        k=1,
    )

    # Experiment 2b: same query, k=3, to show recovery once the right chunk is in
    run(
        label="Experiment 2b — Same query, k=3 (recovery)",
        doc_name="employee_handbook.txt",
        collection="employee_handbook_phase12",
        chunk_size=150,
        overlap=15,
        query="How many consecutive late arrivals trigger a review?",
        k=3,
    )

    # Experiment 3: generation failure (arithmetic over retrieved facts)
    run(
        label="Experiment 3 — Generation failure (arithmetic)",
        doc_name="company_policy.txt",
        collection="company_policy_phase12_genfail",
        chunk_size=100,
        overlap=10,
        query="How many loyalty points would I need to redeem for a $25 discount?",
        k=3,
    )


if __name__ == "__main__":
    main()
