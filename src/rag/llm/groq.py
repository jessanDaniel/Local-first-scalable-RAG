"""Groq implementation of the LLM interface (hosted, free tier, optional)."""

import time

from rag.llm.http import post_json
from rag.llm.interface import LLM, Generation, LLMError, chat_messages

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM(LLM):
    provider = "groq"

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
        temperature: float,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> Generation:
        start = time.perf_counter()
        data = post_json(
            GROQ_CHAT_URL,
            {
                "model": self.model,
                "messages": chat_messages(prompt, system),
                "temperature": self.temperature,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        elapsed = time.perf_counter() - start

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Groq response: {str(data)[:500]}") from exc

        usage = data.get("usage") or {}
        return Generation(
            text=text,
            provider=self.provider,
            model=self.model,
            elapsed_seconds=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
