"""Runtime configuration, read once from environment variables and `.env`.

Every tunable (provider, model names, URLs, timeouts) lives here. Pipeline
code receives values from Settings; it never reads os.environ or hard-codes
a model name.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = PROJECT_ROOT / "data"

    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_pdf_url: str = "https://arxiv.org/pdf"
    # arXiv asks for no more than one request every 3 seconds.
    arxiv_delay_seconds: float = Field(default=3.0, ge=3.0)
    # Generic on purpose: never put personal details (e.g. an email) in request headers.
    http_user_agent: str = "rag-papers-research/0.1"

    grobid_url: str = "http://localhost:8070"
    grobid_timeout_seconds: float = Field(default=180.0, gt=0)
    grobid_max_retries: int = Field(default=3, ge=0)

    llm_provider: Literal["ollama", "groq"] = "ollama"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    groq_api_key: SecretStr | None = None
    groq_model: str | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"

    @model_validator(mode="after")
    def check_provider_settings(self) -> "Settings":
        if self.llm_provider == "groq":
            if self.groq_api_key is None or not self.groq_api_key.get_secret_value():
                raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
            if not self.groq_model:
                raise ValueError("LLM_PROVIDER=groq requires GROQ_MODEL")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
