"""Count tokens the way the embedding model will.

Chunk size is measured in the embedding model's own tokens, not words: the
learning track found that all-MiniLM-L6-v2 silently dropped everything past
token 256, which hid a sentence at token 349 of a 300-word chunk (Phase 14).
"""

from typing import Protocol


class TokenCounter(Protocol):
    max_tokens: int  # the model's hard limit; text past it is never embedded

    def count(self, text: str) -> int: ...


class HFTokenCounter:
    """Token counts from a Hugging Face tokenizer, without [CLS]/[SEP]."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        from transformers.utils import logging

        # Counting a long paragraph is the point here; the "longer than max length" warning is noise.
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_tokens = self.tokenizer.model_max_length

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
