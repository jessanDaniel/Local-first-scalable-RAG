"""Phase 11 — Ollama implementation of the LLM interface.

Same behavior as Phase 10, now wrapped in a class that implements LLM, so the
rest of the pipeline can depend on the interface instead of on Ollama
specifically."""

import time

import requests

from src.llm.interface import LLM

OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaLLM(LLM):
    def __init__(self, model: str = "llama3.2:1b"):
        self.model = model

    def generate(self, prompt: str) -> dict:
        start = time.perf_counter()
        response = requests.post(
            OLLAMA_URL,
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - start

        data = response.json()
        return {
            "answer": data["response"],
            "model": self.model,
            "elapsed_seconds": elapsed,
        }


def print_generation_debug(result: dict) -> None:
    print(f"Model: {result['model']}")
    print(f"Elapsed: {result['elapsed_seconds']:.2f}s")
    print("Answer:")
    print(result["answer"])