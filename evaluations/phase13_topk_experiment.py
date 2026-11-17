"""Phase 13 — Top-K experiments.

Sweep K in {1, 3, 5, 10} across 3 queries (2 documents), measuring:
- retrieval hit (is the fact-bearing chunk actually in the retrieved set?)
- context size (chunks, chars, words as a token proxy)
- retrieval latency
- generation latency
- answer correctness (keyword check against a known ground-truth fact)

Question this answers with evidence: is more context always better?
"""

import csv
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedder import Embedder
from src.generation.context_builder import build_context, build_prompt
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.llm.groq import GroqLLM
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

K_VALUES = [1, 3, 5, 10]

QUERIES = [
    {
        "doc_name": "company_policy.txt",
        "collection": "company_policy_phase13",
        "chunk_size": 100,
        "overlap": 10,
        "query": "How many days do I have to request a refund?",
        "needle": "30 days",
        "expect_in_answer": "30",
    },
    {
        "doc_name": "employee_handbook.txt",
        "collection": "employee_handbook_phase12",  # reuse Phase 12's populated collection
        "chunk_size": 150,
        "overlap": 15,
        "query": "How many consecutive late arrivals trigger a review?",
        "needle": "three late arrivals",
        "expect_in_answer": "three",
    },
    {
        "doc_name": "employee_handbook.txt",
        "collection": "employee_handbook_phase12",
        "chunk_size": 150,
        "overlap": 15,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
]


def generate_with_retry(llm: GroqLLM, prompt: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            return llm.generate(prompt)
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


def main() -> None:
    embedder = Embedder()
    llm = GroqLLM()
    rows = []

    for spec in QUERIES:
        doc = load_txt(project_root / "documents" / spec["doc_name"])
        chunks = chunk_text(doc.document_id, doc.text, chunk_size=spec["chunk_size"], overlap=spec["overlap"])
        n_chunks_total = len(chunks)

        vector_store = VectorStore(persist_path=project_root / "chroma_db", collection_name=spec["collection"])
        retriever = Retriever(vector_store=vector_store, embedder=embedder)
        if vector_store.collection.count() == 0:
            retriever.index_chunks(chunks)

        for k in K_VALUES:
            t0 = time.perf_counter()
            results = retriever.retrieve(spec["query"], k=k)
            retrieval_time = time.perf_counter() - t0

            hit = any(spec["needle"].lower() in r.text.lower() for r in results)
            context = build_context(results)
            prompt = build_prompt(spec["query"], results)

            gen_result = generate_with_retry(llm, prompt)
            time.sleep(1.5)  # be gentle with the free-tier rate limit
            answer = gen_result["answer"]
            correct = spec["expect_in_answer"].lower() in answer.lower()

            row = {
                "doc": spec["doc_name"],
                "query": spec["query"],
                "k": k,
                "n_chunks_total": n_chunks_total,
                "n_chunks_returned": len(results),
                "retrieval_hit": hit,
                "context_chars": len(context),
                "context_words": len(context.split()),
                "retrieval_time_s": round(retrieval_time, 4),
                "generation_time_s": round(gen_result["elapsed_seconds"], 2),
                "answer_correct": correct,
                "answer_preview": answer[:100].replace("\n", " "),
            }
            rows.append(row)
            print(
                f"[{spec['doc_name']:22}] k={k:2} hit={hit!s:5} correct={correct!s:5} "
                f"chunks={row['n_chunks_returned']:2}/{n_chunks_total:2} "
                f"words={row['context_words']:4} gen={row['generation_time_s']:5.2f}s "
                f"| {spec['query'][:45]}"
            )

    out_path = project_root / "evaluations" / "topk_experiment.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
