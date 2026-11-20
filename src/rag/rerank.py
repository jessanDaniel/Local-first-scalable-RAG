"""Cross-encoder reranking: read the question and each candidate chunk together, score the pair.

Dense search is a bi-encoder: question and chunk become vectors separately, then
get compared. A cross-encoder reads "[question] [SEP] [chunk]" as one input, so
every question word can attend to every chunk word: more accurate, but nothing
can be precomputed, so it only reorders a shortlist (hybrid's top 100).

Scores are the model's raw logits (higher = more relevant). sentence-transformers
applies a sigmoid to some models by default (bge-reranker), which squashes
everything near 0 or 1; raw logits keep the gaps readable.

Pairs longer than the model's limit (512 tokens) are never truncated silently:
the "title | section path" header is dropped first, and a pair still too long
is truncated by the model but counted in `last_truncated` (shown in --explain
and recorded in eval runs).
"""

import time


class Reranker:
    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        start = time.perf_counter()
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)
        self.device = str(self.model.device)
        self.max_tokens = self.model.max_length
        self._identity = torch.nn.Identity()
        self.load_seconds = time.perf_counter() - start
        self.last_truncated = 0

    def pair_tokens(self, query: str, passage: str) -> int:
        return len(self.model.tokenizer(query, passage)["input_ids"])

    def fit(self, query: str, header: str, text: str) -> str:
        """The passage to score: header + text if the pair fits, else text alone."""
        with_header = f"{header}\n{text}"
        if self.pair_tokens(query, with_header) <= self.max_tokens:
            return with_header
        if self.pair_tokens(query, text) > self.max_tokens:
            self.last_truncated += 1
        return text

    def score(self, query: str, passages: list[tuple[str, str]]) -> list[float]:
        """passages: (header, text). Returns one raw relevance score per passage."""
        self.last_truncated = 0
        pairs = [(query, self.fit(query, header, text)) for header, text in passages]
        if not pairs:
            return []
        scores = self.model.predict(pairs, batch_size=self.batch_size, activation_fn=self._identity)
        return [float(s) for s in scores]
