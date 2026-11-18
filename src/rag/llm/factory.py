"""Build the configured LLM. The only place that maps LLM_PROVIDER to a class."""

from rag.config import Settings, get_settings
from rag.llm.groq import GroqLLM
from rag.llm.interface import LLM
from rag.llm.ollama import OllamaLLM


def create_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()

    if settings.llm_provider == "ollama":
        return OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            num_ctx=settings.ollama_num_ctx,
            max_tokens=settings.llm_max_answer_tokens,
        )
    if settings.llm_provider == "groq":
        return GroqLLM(
            model=settings.groq_model,
            api_key=settings.groq_api_key.get_secret_value(),
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            context_window=settings.groq_context_window,
            max_tokens=settings.llm_max_answer_tokens,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
