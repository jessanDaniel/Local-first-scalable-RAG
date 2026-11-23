"""Ollama implementation of the LLM interface (local, $0)."""

import time
from collections.abc import Callable
from dataclasses import replace

from rag.llm.http import post_json, post_json_lines
from rag.llm.interface import LLM, Generation, LLMError, chat_messages


def nanoseconds(value) -> float | None:
    """Ollama reports durations in nanoseconds."""
    return value / 1e9 if isinstance(value, int | float) else None


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        temperature: float,
        num_ctx: int,
        max_tokens: int,
    ):
        self.model = model
        self.context_window = num_ctx
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature

    def payload(self, prompt: str, system: str | None, stream: bool) -> dict:
        """One request body for both paths, so streaming and not streaming ask the same thing."""
        return {
            "model": self.model,
            "messages": chat_messages(prompt, system),
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_window,
                "num_predict": self.max_tokens,
            },
        }

    def generate(self, prompt: str, system: str | None = None) -> Generation:
        start = time.perf_counter()
        data = post_json(
            f"{self.base_url}/api/chat",
            self.payload(prompt, system, stream=False),
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        elapsed = time.perf_counter() - start

        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:500]}") from exc
        return self.generation(text, data, elapsed)

    def generate_stream(self, prompt: str, system: str | None, on_token: Callable[[str], None]) -> Generation:
        """Ollama streams one JSON line per piece ({"message": {"content": ...}, "done": false});
        the last line has done=true and the token counts and durations."""
        start = time.perf_counter()
        pieces: list[str] = []
        final: dict | None = None
        first: float | None = None
        lines = post_json_lines(
            f"{self.base_url}/api/chat",
            self.payload(prompt, system, stream=True),
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        try:
            for data in lines:
                if "error" in data:
                    raise LLMError(f"Ollama error: {data['error']}")
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    if first is None:
                        first = time.perf_counter() - start
                    pieces.append(piece)
                    on_token(piece)  # may raise GenerationCancelled; closing `lines` ends the request
                if data.get("done"):
                    final = data
                    break
        finally:
            lines.close()
        if final is None:
            raise LLMError("Ollama stream ended without a final message")
        generation = self.generation("".join(pieces), final, time.perf_counter() - start)
        return replace(generation, first_token_seconds=first)

    def generation(self, text: str, data: dict, elapsed: float) -> Generation:
        return Generation(
            text=text,
            provider=self.provider,
            model=self.model,
            elapsed_seconds=elapsed,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            prompt_seconds=nanoseconds(data.get("prompt_eval_duration")),
            write_seconds=nanoseconds(data.get("eval_duration")),
            load_seconds=nanoseconds(data.get("load_duration")),
        )
