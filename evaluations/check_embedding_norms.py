"""Phase 7 step 1 — Empirically check whether chunk embeddings are unit-length.

Embedder.embed_texts() calls model.encode(..., convert_to_numpy=True) with no
normalize_embeddings=True, so we expect norms to vary chunk-to-chunk, not all
equal 1.0. This grounds the L2-vs-cosine discussion in real numbers."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_txt


def main() -> None:
    doc = load_txt(project_root / "documents" / "company_policy.txt")
    chunks = chunk_text(doc.document_id, doc.text, chunk_size=100, overlap=10)

    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)

    print(f"{'chunk_id':<20}{'word_count':>12}{'norm':>12}")
    for ec in embedded:
        norm = np.linalg.norm(ec.embedding)
        word_count = len(ec.chunk.text.split())
        print(f"{ec.chunk.chunk_id:<20}{word_count:>12}{norm:>12.4f}")


if __name__ == "__main__":
    main()