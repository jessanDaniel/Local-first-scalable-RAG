from dataclasses import dataclass


@dataclass
class Chunk:
    document_id: str
    chunk_id: str
    position: int
    text: str


def chunk_text(document_id: str, text: str, chunk_size: int = 500, overlap: int = 0) -> list[Chunk]:
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap must be < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")

    words = text.split()

    if not words:
        return []

    chunks: list[Chunk] = []
    stepSizeWithOverlap = chunk_size - overlap

    for i in range(0, len(words), stepSizeWithOverlap):
        chunk_words = words[i : i + chunk_size]
        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)
        position = len(chunks)
        chunk_id = f"{document_id}_{position}"
        chunks.append(Chunk(document_id=document_id, chunk_id=chunk_id, position=position, text=chunk_text))

        if i + chunk_size >= len(words):
            break
    return chunks
