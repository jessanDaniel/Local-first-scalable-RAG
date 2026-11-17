"""Phase 11 — the LLM provider interface.
Anything implementing this can be swapped in as the pipeline's generation
backend with zero changes to retrieval, chunking, or context construction
code. Concrete providers (Ollama, Gemini, Groq, ...) live in their own files
and implement this contract."""

from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> dict:
        """Generate a response for the given prompt.
        Returns a dict with at least: answer, model, elapsed_seconds.
        """
        raise NotImplementedError