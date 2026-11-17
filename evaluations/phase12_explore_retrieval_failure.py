"""Phase 12 exploration — try several candidate queries against employee_handbook
to find a genuine retrieval-failure case (answer exists in the doc, but the
right chunk isn't actually retrieved at k=3). Throwaway diagnostic script."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


def main() -> None:
    doc = load_txt(project_root / "documents" / "employee_handbook.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=150, overlap=15)
    print(f"n_chunks: {len(chunks)}")

    embedder = Embedder()
    vector_store = VectorStore(
        persist_path=project_root / "chroma_db",
        collection_name="employee_handbook_phase12",
    )
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    retriever.index_chunks(chunks)

    candidates = [
        ("What is the maximum gift value I can accept from a supplier?", "75 dollars"),
        ("Can I keep my personal cloud storage backups of source code?", "personal cloud accounts"),
        ("How many consecutive late arrivals trigger a review?", "three late arrivals"),
        ("What happens to unlabeled food in the office fridge?", "discarded every Friday"),
        ("How long before a promotion is blocked if I skip training?", "blocks promotion eligibility"),
        ("Is alcohol reimbursable on a business trip?", "Alcohol is reimbursable"),
        ("What badge access logging happens after hours?", "logged automatically"),
        ("How much notice do managers need to give when resigning?", "four weeks"),
    ]

    for query, needle in candidates:
        results = retriever.retrieve(query, k=3)
        found_ranks = [r.rank for r in results if needle.lower() in r.text.lower()]
        print(f"\nQuery: {query}")
        print(f"  Looking for: {needle!r}")
        print(f"  Found at rank(s): {found_ranks if found_ranks else 'NOT IN TOP 3'}")
        for r in results:
            preview = r.text[:80].replace("\n", " ")
            print(f"    rank={r.rank} score={r.score:.4f} chunk={r.chunk_id} text=\"{preview}...\"")


if __name__ == "__main__":
    main()
