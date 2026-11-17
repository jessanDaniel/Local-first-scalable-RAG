"""Phase 14 — chunking × Top-K matrix (retrieval only).
Same three Phase 13 queries, swept across chunk size, overlap, and K.
No LLM call. Records whether the fact-bearing text was retrieved, at what
rank, and how large the context was.
"""

import csv
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.embeddings.embedder import Embedder
from src.generation.context_builder import build_context
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

K_VALUES = [1, 3, 5, 10]
QUERIES = [
    {
        "doc_name": "company_policy.txt",
        "query": "How many days do I have to request a refund?",
        "needle": "30 days",
    },
    {
        "doc_name": "employee_handbook.txt",
        "query": "How many consecutive late arrivals trigger a review?",
        "needle": "three late arrivals",
    },
    {
        "doc_name": "employee_handbook.txt",
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
    },
]

# (chunk_size, overlap) per document. Overlap is a word count, ~10% of size.
# The second policy config and the second handbook config match Phase 13.
CONFIGS = {
    "company_policy.txt": [(100, 0), (100, 10), (200, 0), (200, 20)],
    "employee_handbook.txt": [(150, 0), (150, 15), (300, 0), (300, 30)],
}

def collection_name(doc_name: str, chunk_size: int, overlap: int) -> str:
    stem = Path(doc_name).stem
    return f"p14_{stem}_{chunk_size}_{overlap}"



def needle_rank(results, needle: str) -> int | None:
    needle = needle.lower()
    for result in results:
        if needle in result.text.lower():
            return result.rank
    return None


def main() -> None:
    embedder = Embedder()
    client_path = project_root / "chroma_db"
    rows = []
    for doc_name, configs in CONFIGS.items():
        doc = load_txt(project_root / "documents" / doc_name)
        doc_queries = [q for q in QUERIES if q["doc_name"] == doc_name]

        for chunk_size, overlap in configs:
            chunks = chunk_text(
                doc.document_id, doc.text, chunk_size=chunk_size, overlap=overlap
            )
            name = collection_name(doc_name, chunk_size, overlap)

            # Fresh index every run. get_or_create would keep a previous cut.
            store = VectorStore(persist_path=client_path, collection_name=name)
            store.client.delete_collection(name)
            store = VectorStore(persist_path=client_path, collection_name=name)
            retriever = Retriever(vector_store=store, embedder=embedder)
            retriever.index_chunks(chunks)

            # Needle absent from every chunk => boundary split, not a ranking miss.
            needle_in_index = {
                q["query"]: any(q["needle"].lower() in c.text.lower() for c in chunks)
                for q in doc_queries
            }

            for spec in doc_queries:
                for k in K_VALUES:
                    results = retriever.retrieve(spec["query"], k=k)
                    rank = needle_rank(results, spec["needle"])
                    context = build_context(results)
                    row = {
                        "doc": doc_name,
                        "query": spec["query"],
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "k": k,
                        "n_chunks_total": len(chunks),
                        "n_chunks_returned": len(results),
                        "needle_in_index": needle_in_index[spec["query"]],
                        "retrieval_hit": rank is not None,
                        "needle_rank": "" if rank is None else rank,
                        "top_score": round(results[0].score, 4) if results else "",
                        "context_words": len(context.split()),
                    }

                    rows.append(row)
                    print(
                        f"[{doc_name:22}] size={chunk_size:3} ov={overlap:2} k={k:2} "
                        f"hit={row['retrieval_hit']!s:5} rank={str(row['needle_rank']):4} "
                        f"in_index={row['needle_in_index']!s:5} "
                        f"chunks={row['n_chunks_returned']:2}/{len(chunks):2} "
                        f"words={row['context_words']:4}"
                    )

    out_path = project_root / "evaluations" / "phase14_chunk_topk_matrix.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

