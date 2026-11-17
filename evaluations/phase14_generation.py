"""Phase 14 — generation on the cells where retrieval rank changed.
Nine Groq calls. Keyword check is a first pass only; read answer_preview
before trusting answer_correct. A hit of "15" inside "150" does not count.
"""
import csv
import re
import sys
import time
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.embeddings.embedder import Embedder
from src.generation.context_builder import build_prompt
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt
from src.llm.groq import GroqLLM
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

CELLS = [
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 150,
        "overlap": 0,
        "k": 1,
        "query": "How many consecutive late arrivals trigger a review?",
        "needle": "three late arrivals",
        "expect_in_answer": "three",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 300,
        "overlap": 0,
        "k": 5,
        "query": "How many consecutive late arrivals trigger a review?",
        "needle": "three late arrivals",
        "expect_in_answer": "three",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 300,
        "overlap": 0,
        "k": 10,
        "query": "How many consecutive late arrivals trigger a review?",
        "needle": "three late arrivals",
        "expect_in_answer": "three",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 150,
        "overlap": 0,
        "k": 1,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 150,
        "overlap": 0,
        "k": 10,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 300,
        "overlap": 0,
        "k": 1,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 300,
        "overlap": 0,
        "k": 10,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 300,
        "overlap": 30,
        "k": 3,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
    {
        "doc_name": "employee_handbook.txt",
        "chunk_size": 300,
        "overlap": 30,
        "k": 10,
        "query": "How many vacation days do employees get in their first year of service?",
        "needle": "15 days of paid vacation in years",
        "expect_in_answer": "15",
    },
]

def collection_name(doc_name: str, chunk_size: int, overlap: int) -> str:
    stem = Path(doc_name).stem
    return f"p14_{stem}_{chunk_size}_{overlap}"

def answer_has(answer: str, expected: str) -> bool:
    return re.search(rf"\b{re.escape(expected)}\b", answer, re.IGNORECASE) is not None

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
    client_path = project_root / "chroma_db"
    rows = []
    indexed: set[str] = set()
    for spec in CELLS:
        doc = load_txt(project_root / "documents" / spec["doc_name"])
        chunks = chunk_text(
            doc.document_id,
            doc.text,
            chunk_size=spec["chunk_size"],
            overlap=spec["overlap"],
        )
        name = collection_name(spec["doc_name"], spec["chunk_size"], spec["overlap"])
        store = VectorStore(persist_path=client_path, collection_name=name)
        retriever = Retriever(vector_store=store, embedder=embedder)
        if name not in indexed:
            if store.collection.count() != len(chunks):
                raise RuntimeError(
                    f"{name} has {store.collection.count()} chunks, expected {len(chunks)}. "
                    "Re-run phase14_chunk_topk_matrix.py first."
                )
            indexed.add(name)
        
        results = retriever.retrieve(spec["query"], k=spec["k"])
        hit = any(spec["needle"].lower() in r.text.lower() for r in results)
        prompt = build_prompt(spec["query"], results)
        gen = generate_with_retry(llm, prompt)
        time.sleep(1.5)
        answer = gen["answer"]
        correct = answer_has(answer, spec["expect_in_answer"])

        row = {
            "doc": spec["doc_name"],
            "query": spec["query"],
            "chunk_size": spec["chunk_size"],
            "overlap": spec["overlap"],
            "k": spec["k"],
            "retrieval_hit": hit,
            "context_words": len(prompt.split()),
            "answer_correct": correct,
            "generation_time_s": round(gen["elapsed_seconds"], 2),
            "answer_preview": answer[:180].replace("\n", " "),
        }

        rows.append(row)
        print(
            f"size={spec['chunk_size']:3} ov={spec['overlap']:2} k={spec['k']:2} "
            f"hit={hit!s:5} correct={correct!s:5} | {answer[:120].replace(chr(10), ' ')}"
        )
    out_path = project_root / "evaluations" / "phase14_generation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")
if __name__ == "__main__":
    main()
