"""Text -> unit-length vectors with a sentence-transformers model (bge-small-en-v1.5).

- Passages are embedded as "<title> | <section path>\\n<chunk text>": the header
  gives short chunks ("we set it to 0.1") the context of their paper and section.
  It is trimmed to HEADER_TOKENS so header + chunk always fit the model.
- Queries get the model's retrieval instruction prepended. bge is trained that
  way; keeping it here means no caller can forget it.
- Nothing is silently truncated: every input is counted and an input over the
  model's limit raises InputTooLongError (the bug that hid a sentence in Phase 14).
- Vectors are L2-normalized, so inner product == cosine similarity.
"""

import time

import numpy as np

from rag.chunk.chunker import HEADER_RESERVE_TOKENS
from rag.chunk.tokens import TokenCounter

HEADER_SEPARATOR = " | "
HEADER_TOKENS = HEADER_RESERVE_TOKENS - 4  # margin for the newline and special tokens
PATH_SEPARATOR = " > "


class InputTooLongError(ValueError):
    """An embedding input is longer than the model reads; it would be cut off silently."""


def passage_text(title: str, section_path: str, chunk_text: str, counter: TokenCounter) -> str:
    """The exact string embedded for a chunk: a header within HEADER_TOKENS, then the chunk."""
    return f"{trim_header(title, section_path, counter)}\n{chunk_text}"


def trim_header(title: str, section_path: str, counter: TokenCounter) -> str:
    """Shorten "<title> | <path>" to HEADER_TOKENS: drop outer path levels first (the innermost
    section says the most about the chunk), then cut the title's tail."""
    parts = [p for p in section_path.split(PATH_SEPARATOR) if p]

    def header(title_words: list[str], path_parts: list[str]) -> str:
        return HEADER_SEPARATOR.join(x for x in (" ".join(title_words), PATH_SEPARATOR.join(path_parts)) if x)

    title_words = title.split()
    while len(parts) > 1 and counter.count(header(title_words, parts)) > HEADER_TOKENS:
        parts = parts[1:]
    while len(title_words) > 1 and counter.count(header(title_words, parts)) > HEADER_TOKENS:
        title_words = title_words[:-1]
    text = header(title_words, parts)
    while counter.count(text) > HEADER_TOKENS:  # one enormous heading
        text = text[: len(text) * 3 // 4]
    return text


class Embedder:
    def __init__(
        self,
        model_name: str,
        query_instruction: str,
        batch_size: int = 64,
        device: str | None = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.max_tokens = self.model.max_seq_length
        self.dim = self.model.get_embedding_dimension()
        self.last_seconds = 0.0

    def count(self, text: str) -> int:
        """Tokens the model sees for this text, including [CLS] and [SEP]."""
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([f"{self.query_instruction}{query}"])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        for text in texts:
            n = self.count(text)
            if n > self.max_tokens:
                raise InputTooLongError(
                    f"{n} tokens > {self.max_tokens} the model reads; the end would be dropped: {text[:80]!r}"
                )
        start = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.last_seconds = time.perf_counter() - start
        return np.asarray(vectors, dtype=np.float32)
