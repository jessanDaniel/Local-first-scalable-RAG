"""Phase 3 — Inspect what happens when text crosses a chunk boundary."""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.ingestion.chunker import chunk_text
from rag.ingestion.loader import load_txt


def main() -> None:
    doc = load_txt(project_root / "documents" / "employee_handbook.txt")
    words = doc.text.split()

    # Pick a boundary that falls in the middle of the doc for chunk_size=200
    chunk_size = 200
    boundary = chunk_size  # words[199] ends chunk 0; words[200] starts chunk 1

    # A short phrase that sits across that boundary
    probe = " ".join(words[boundary - 5 : boundary + 5])
    print(f"Probe phrase spanning the boundary:\n  ...{probe}...\n")

    for overlap in (0, 40):  # 0% and 20% of 200
        chunks = chunk_text(
            doc.document_id,
            doc.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        print(f"--- chunk_size={chunk_size}, overlap={overlap} ---")
        hits = [c for c in chunks if probe in c.text]
        if hits:
            for c in hits:
                print(f"  FOUND whole probe in {c.chunk_id}")
        else:
            print("  Probe NOT intact in any single chunk")
            # Show which chunks hold the left/right halves
            left = " ".join(words[boundary - 5 : boundary])
            right = " ".join(words[boundary : boundary + 5])
            for c in chunks:
                has_left = left in c.text
                has_right = right in c.text
                if has_left or has_right:
                    print(
                        f"  {c.chunk_id}: left_half={has_left}, "
                        f"right_half={has_right}"
                    )
        print()


if __name__ == "__main__":
    main()