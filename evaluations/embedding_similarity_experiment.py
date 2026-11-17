import sys

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.embeddings.embedder import Embedder

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare(embedder: Embedder, label: str, text_a: str, text_b: str):
    va = embedder.embed_query(text_a)
    vb = embedder.embed_query(text_b)
    similarity = cosine_similarity(va, vb)
    print(f"[{label}]")
    print(f"  A: {text_a}")
    print(f"  B: {text_b}")
    print(f"  cosine similarity: {similarity:.4f}")
    print()

def main() -> None:
    embedder = Embedder()

    compare(embedder, "A — paraphrase (should be HIGH)",
        "The dog is sleeping.",
        "A puppy is taking a nap.",
    )

    compare(embedder, 
    "B — unrelated (should be LOW)",
        "The dog is sleeping.",
        "The stock market crashed.",
    )

    compare(
        embedder,
        "C — ambiguous word 'Java'",
        "Java is a programming language.",
        "Java is an island.",
    )


if __name__ == "__main__":
    main()