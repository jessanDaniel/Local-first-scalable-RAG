"""Phase 16 — Hit@K and Recall@K on the Phase 15 question set.

Chunking is fixed: company_policy 100/10, employee_handbook 150/15.
No LLM call. Missing-information rows are skipped (no source span).
"""

import csv
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.embeddings.embedder import Embedder
from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore

K_VALUES = [1, 3, 5, 10]
CHUNKING = {
    "company_policy.txt": (100, 10),
    "employee_handbook.txt": (150, 15),
}


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def first_rank(results, quote: str) -> int | None:
    needle = normalize(quote)
    for result in results:
        if needle in normalize(result.text):
            return result.rank
    return None


def collection_name(doc_name: str, chunk_size: int, overlap: int) -> str:
    stem = Path(doc_name).stem
    return f"p16_{stem}_{chunk_size}_{overlap}"


def main() -> None:
    questions = [
        json.loads(line)
        for line in (project_root / "experiments" / "learning" / "eval_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    embedder = Embedder()
    client_path = project_root / "chroma_db"
    rows = []

    for doc_name, (chunk_size, overlap) in CHUNKING.items():
        doc = load_txt(project_root / "documents" / doc_name)
        chunks = chunk_text(doc.document_id, doc.text, chunk_size=chunk_size, overlap=overlap)
        name = collection_name(doc_name, chunk_size, overlap)
        store = VectorStore(persist_path=client_path, collection_name=name)
        store.client.delete_collection(name)
        store = VectorStore(persist_path=client_path, collection_name=name)
        retriever = Retriever(vector_store=store, embedder=embedder)
        retriever.index_chunks(chunks)

        doc_questions = [
            q for q in questions
            if q["doc"] == doc_name and q["type"] != "missing_information"
        ]
        for spec in doc_questions:
            quotes = spec["source_quotes"]
            intact = sum(
                1 for quote in quotes
                if any(normalize(quote) in normalize(chunk.text) for chunk in chunks)
            )
            for k in K_VALUES:
                results = retriever.retrieve(spec["question"], k=k)
                source_ranks = [first_rank(results, quote) for quote in quotes]
                found = [rank for rank in source_ranks if rank is not None]
                distractor_ranks = [
                    first_rank(results, quote)
                    for quote in spec["distractor_quotes"]
                ]
                distractor_found = [rank for rank in distractor_ranks if rank is not None]
                best_source = min(found) if found else None
                best_distractor = min(distractor_found) if distractor_found else None
                outranks = best_distractor is not None and (
                    best_source is None or best_distractor < best_source
                )
                row = {
                    "id": spec["id"],
                    "doc": doc_name,
                    "type": spec["type"],
                    "k": k,
                    "n_quotes": len(quotes),
                    "quotes_intact": intact,
                    "quotes_found": len(found),
                    "hit": len(found) > 0,
                    "recall": round(len(found) / len(quotes), 4),
                    "complete": len(found) == len(quotes),
                    "best_source_rank": "" if best_source is None else best_source,
                    "distractor_retrieved": best_distractor is not None,
                    "distractor_outranks_source": outranks,
                }
                rows.append(row)

    out_path = project_root / "experiments" / "learning" / "phase16_retrieval_eval.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")
    print(f"{'type':22} {'k':>2} {'n':>3} {'hit':>6} {'recall':>7} {'complete':>8} {'distractor':>10}")
    types = ["direct", "multi_step", "distractor"]
    for qtype in types + ["ALL"]:
        for k in K_VALUES:
            subset = [
                row for row in rows
                if row["k"] == k and (qtype == "ALL" or row["type"] == qtype)
            ]
            n = len(subset)
            hit = sum(row["hit"] for row in subset) / n
            recall = sum(row["recall"] for row in subset) / n
            complete = sum(row["complete"] for row in subset) / n
            distractor = sum(row["distractor_retrieved"] for row in subset) / n
            print(
                f"{qtype:22} {k:2} {n:3} {hit:6.2f} {recall:7.2f} {complete:8.2f} {distractor:10.2f}"
            )


if __name__ == "__main__":
    main()