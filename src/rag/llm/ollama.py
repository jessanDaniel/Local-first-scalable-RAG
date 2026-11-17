"""Ollama implementation of the LLM interface (local, $0)."""

import time

from rag.llm.http import post_json
from rag.llm.interface import LLM, Generation, LLMError, chat_messages


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        temperature: float,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> Generation:
        start = time.perf_counter()
        data = post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": chat_messages(prompt, system),
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        elapsed = time.perf_counter() - start

        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:500]}") from exc

        return Generation(
            text=text,
            provider=self.provider,
            model=self.model,
            elapsed_seconds=elapsed,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
