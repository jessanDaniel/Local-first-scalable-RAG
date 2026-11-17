
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.ingestion.chunker import Chunk

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: np.ndarray

class Embedder:
    def __init__(self, model_name:str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return np.asarray(vectors)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
    
    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts)

        return [EmbeddedChunk(chunk=chunk, embedding=vector) for chunk, vector in zip(chunks, vectors)]

