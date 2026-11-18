"""The LLM provider contract.

The pipeline depends on LLM and Generation only. Concrete providers live in
their own modules; rag.llm.factory picks one from Settings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """A provider call failed after retries, or returned an unusable response."""


@dataclass(frozen=True)
class Generation:
    text: str
    provider: str
    model: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class LLM(ABC):
    provider: str
    model: str
    context_window: int  # prompt + answer tokens the model accepts per call
    max_tokens: int  # longest answer requested

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> Generation:
        raise NotImplementedError


def chat_messages(prompt: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def print_generation_debug(generation: Generation) -> None:
    print(f"Provider: {generation.provider}")
    print(f"Model: {generation.model}")
    print(f"Elapsed: {generation.elapsed_seconds:.2f}s")
    print(
        f"Tokens: prompt={generation.prompt_tokens} "
        f"completion={generation.completion_tokens} total={generation.total_tokens}"
    )
    print("Answer:")
    print(generation.text)
