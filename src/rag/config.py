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

    # PDF parser (step 21b): "pymupdf" = rag.parse.pdf, no service, any kind of PDF (the default);
    # "grobid" = GROBID for scientific papers (needs the GROBID server; falls back to pymupdf on a
    # parse error). Override per run with `rag ingest ... --parser`.
    pdf_parser: Literal["pymupdf", "grobid"] = "pymupdf"

    grobid_url: str = "http://localhost:8070"
    grobid_timeout_seconds: float = Field(default=180.0, gt=0)
    grobid_max_retries: int = Field(default=3, ge=0)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Prepended to queries only (bge's documented instruction for short query -> passage search).
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=64, ge=1)
    # Chunk text budget, in embedding-model tokens. Kept well under the model's 512 so the
    # contextual header ("title | section path") added at embedding time still fits.
    chunk_max_tokens: int = Field(default=350, ge=32)

    llm_provider: Literal["ollama", "groq"] = "ollama"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    # Longest answer the model may write, in its own tokens.
    llm_max_answer_tokens: int = Field(default=600, ge=16)
    # Tokenizer used to count prompt tokens before sending (should match the LLM's).
    llm_tokenizer: str = "Qwen/Qwen2.5-7B-Instruct"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    # Ollama's context window. Set explicitly: Ollama's default is smaller than the model's, and a
    # longer prompt is cut from the start without an error.
    ollama_num_ctx: int = Field(default=8192, ge=512)

    groq_api_key: SecretStr | None = None
    groq_model: str | None = None
    groq_context_window: int = Field(default=8192, ge=512)

    # Answering: how many hits to retrieve, how many sources at most, and the token budget for
    # the sources block of the prompt.
    # Retrieval: which retriever `rag ask` uses, and the hybrid settings (Stage 2 step 14).
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    hybrid_candidates: int = Field(default=100, ge=1)  # each retriever's list size before fusion
    rrf_k: int = Field(default=60, ge=0)  # Reciprocal Rank Fusion constant (Cormack et al. 2009)
    # Reranking on top of the retriever (step 15b): off | fused (the reranker votes in RRF) |
    # replace (its order is final). "off" never loads the model. Override per command with --rerank.
    rerank: Literal["off", "fused", "replace"] = "off"  # default for rag search and rag eval
    # `rag ask` adds generation (6-13 s), so the reranker's ~1.8 s is a small share there. Fused won
    # the step-15b decision rule (papers Hit@1 26 vs 22, MRR 0.755 vs 0.675;
    # experiments/stage2/rerank_fused_step15b.md).
    ask_rerank: Literal["off", "fused", "replace"] = "fused"
    # Answer prompt (rag/answer/prompt.py) and post-generation guardrails (rag/answer/guardrails.py,
    # step 17: flag, don't block; refusal citations are stripped).
    ask_prompt: Literal["v2", "v3"] = "v3"
    ask_guardrails: bool = True
    # Remove instruction-like sentences from sources before generation (prompt-injection defence).
    # On by the user's decision (2026-11-23): relevant injections followed 4/18 -> 0/18, 0/429 real
    # chunks affected; held-out recall 7/12, so the flags stay on too (experiments/stage2/guardrails_step17.md).
    ask_sanitize_sources: bool = True
    # Cross-encoder reranking (Stage 2 step 15): the retriever's top
    # `rerank_candidates` are rescored and the best `rerank_keep` kept. Both models were approved by the
    # user; bge measured better than ms-marco-MiniLM-L-6-v2 (papers Hit@5 37 vs 34, MRR 0.695 vs 0.661)
    # but costs ~1.8 s per query, so retrieval_mode stays hybrid (experiments/stage2/rerank_step15.md).
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = Field(default=100, ge=1)
    rerank_keep: int = Field(default=20, ge=1)
    rerank_batch_size: int = Field(default=32, ge=1)
    rerank_timeout_seconds: float = Field(default=10.0, gt=0)

    ask_retrieve_k: int = Field(default=10, ge=1)
    ask_max_sources: int = Field(default=8, ge=1)
    ask_context_tokens: int = Field(default=3000, ge=200)

    # API server (`rag serve`, Stage 3 step 19). Loopback only until API keys exist (step 29).
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # Request logs (step 20): one JSON line per `rag search` / `rag ask` / API request in
    # data/logs/requests.jsonl, read with `rag logs show|stats`. LOG_TEXT stores the question, the
    # answer and the guardrail flag text; on while developing (the user's choice, 2026-11-23), to be
    # revisited for the packaged release.
    log_requests: bool = True
    log_text: bool = True
    log_max_mb: float = Field(default=20.0, gt=0)
    log_backups: int = Field(default=5, ge=1)

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
