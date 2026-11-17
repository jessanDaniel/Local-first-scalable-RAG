"""Phase 11 — Groq implementation of the LLM interface (free tier, hosted).

Same contract as OllamaLLM: construct with a model, call generate(prompt).
This is the actual proof that the interface works: swapping this in for
OllamaLLM anywhere in the pipeline requires zero changes to retrieval,
chunking, or context construction code."""

import os
import time

import requests
from dotenv import load_dotenv

from rag.llm.interface import LLM

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM(LLM):
    def __init__(self, model: str | None = None):
        self.model = model or os.environ["GROQ_MODEL"]
        self.api_key = os.environ["GROQ_API_KEY"]

    def generate(self, prompt: str) -> dict:
        start = time.perf_counter()
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - start

        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return {
            "answer": answer,
            "model": self.model,
            "elapsed_seconds": elapsed,
        }