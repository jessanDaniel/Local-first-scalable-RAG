import pytest
from pydantic import ValidationError

from rag.config import Settings
from rag.llm import groq, ollama
from rag.llm.factory import create_llm
from rag.llm.groq import GroqLLM
from rag.llm.http import backoff_seconds
from rag.llm.interface import Generation, LLMError
from rag.llm.ollama import OllamaLLM


def make_settings(**overrides) -> Settings:
    # _env_file=None: tests must not depend on the developer's .env
    return Settings(_env_file=None, **overrides)


def test_defaults_to_ollama():
    settings = make_settings()
    assert settings.llm_provider == "ollama"
    assert isinstance(create_llm(settings), OllamaLLM)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="openai")


def test_groq_without_key_is_rejected():
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make_settings(llm_provider="groq", groq_model="some-model")


def test_groq_without_model_is_rejected():
    with pytest.raises(ValidationError, match="GROQ_MODEL"):
        make_settings(llm_provider="groq", groq_api_key="key")


def test_env_var_selects_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    llm = create_llm(make_settings())
    assert isinstance(llm, GroqLLM)
    assert llm.model == "some-model"


def test_ollama_response_becomes_generation(monkeypatch):
    def fake_post_json(url, payload, **kwargs):
        assert url == "http://localhost:11434/api/chat"
        assert payload["messages"][0] == {"role": "system", "content": "be brief"}
        return {"message": {"content": "hi"}, "prompt_eval_count": 12, "eval_count": 3}

    monkeypatch.setattr(ollama, "post_json", fake_post_json)
    result = create_llm(make_settings()).generate("hello", system="be brief")
    assert result == Generation(
        text="hi", provider="ollama", model="qwen2.5:7b-instruct",
        elapsed_seconds=result.elapsed_seconds, prompt_tokens=12, completion_tokens=3,
    )
    assert result.total_tokens == 15


def test_groq_response_becomes_generation(monkeypatch):
    def fake_post_json(url, payload, **kwargs):
        assert kwargs["headers"] == {"Authorization": "Bearer key"}
        return {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }

    monkeypatch.setattr(groq, "post_json", fake_post_json)
    llm = create_llm(make_settings(llm_provider="groq", groq_api_key="key", groq_model="m"))
    result = llm.generate("hello")
    assert (result.text, result.prompt_tokens, result.total_tokens) == ("hi", 20, 25)


def test_malformed_response_raises_llm_error(monkeypatch):
    monkeypatch.setattr(ollama, "post_json", lambda *a, **k: {"error": "boom"})
    with pytest.raises(LLMError, match="Unexpected Ollama response"):
        create_llm(make_settings()).generate("hello")


def test_missing_token_counts_give_none():
    result = Generation("hi", "ollama", "m", 0.1, prompt_tokens=None, completion_tokens=3)
    assert result.total_tokens is None


def test_backoff_prefers_retry_after_and_caps():
    assert backoff_seconds(0, "7") == 7.0
    assert backoff_seconds(0, "not-a-number") == 1.0
    assert backoff_seconds(3, None) == 8.0
    assert backoff_seconds(10, None) == 30.0
