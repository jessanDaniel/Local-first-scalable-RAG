"""The loaded system: every expensive part loaded at most once, on first use.

    pipeline = Pipeline()
    pipeline.searcher("hybrid", rerank="fused").search("why sqrt d_k?", 5)
    pipeline.answerer("hybrid", rerank="fused").ask("why sqrt d_k?")

The CLI builds one per command (and throws it away); the API server builds one at startup and
keeps it, which is the whole point of the server: the embedding model, BM25 index, FAISS index,
reranker and prompt tokenizer are loaded once instead of on every request. Both go through this
class, so `rag ask` and `POST /ask` cannot drift apart.

Searchers and answerers are cheap wrappers around the shared parts and are cached per
(mode, rerank). They keep per-call state (last_ranks, last_timings), so a Pipeline must be used
from one thread at a time; the server gives it a dedicated worker thread.
"""

from rag.config import Settings, get_settings
from rag.store import db

MODES = ("hybrid", "dense", "bm25")
RERANK_STRATEGIES = ("off", "fused", "replace")


class Pipeline:
    def __init__(self, settings: Settings | None = None, conn=None, llm=None, prompt_counter=None):
        """conn, llm and prompt_counter can be passed in (tests); otherwise built from settings."""
        self.settings = settings or get_settings()
        self.conn = conn if conn is not None else db.open_db(self.settings.db_path)
        self._llm = llm
        self._dense = None
        self._bm25 = None
        self._reranker = None
        self._prompt_counter = prompt_counter
        self._searchers: dict[tuple[str, str], object] = {}
        self._answerers: dict[tuple[str, str], object] = {}

    # --- the shared parts -------------------------------------------------------------------

    def dense(self):
        """bge-small + the CURRENT FAISS build. Raises StaleIndexError if the index doesn't match
        the database."""
        if self._dense is None:
            from rag.embed.embedder import Embedder
            from rag.index.dense import load_current
            from rag.search import DenseSearcher

            s = self.settings
            index = load_current(s.data_dir / "indexes" / "dense", s.embedding_model, self.conn)
            embedder = Embedder(s.embedding_model, s.embedding_query_instruction, s.embedding_batch_size)
            self._dense = DenseSearcher(self.conn, index, embedder)
        return self._dense

    def bm25(self):
        """Built in memory from SQLite (nothing on disk, never stale)."""
        if self._bm25 is None:
            from rag.search import BM25Searcher

            self._bm25 = BM25Searcher.from_db(self.conn)
        return self._bm25

    def reranker(self):
        """The cross-encoder (~1 GB). Imported only when a rerank strategy other than off is used."""
        if self._reranker is None:
            from rag.rerank import Reranker

            self._reranker = Reranker(self.settings.rerank_model, self.settings.rerank_batch_size)
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from rag.llm.factory import create_llm

            self._llm = create_llm(self.settings)
        return self._llm

    def prompt_counter(self):
        if self._prompt_counter is None:
            from rag.chunk.tokens import HFTokenCounter

            self._prompt_counter = HFTokenCounter(self.settings.llm_tokenizer)
        return self._prompt_counter

    def loaded(self) -> list[str]:
        """Which parts are in memory now (for /health)."""
        parts = {
            "dense": self._dense,
            "bm25": self._bm25,
            "reranker": self._reranker,
            "prompt_tokenizer": self._prompt_counter,
        }
        return [name for name, part in parts.items() if part is not None]

    # --- what callers use -------------------------------------------------------------------

    def searcher(self, mode: str, rerank: str = "off"):
        """The retriever for a mode, optionally wrapped by the reranker; built once per pair."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if rerank not in RERANK_STRATEGIES:
            raise ValueError(f"rerank must be one of {RERANK_STRATEGIES}, not {rerank!r}")
        key = (mode, rerank)
        if key not in self._searchers:
            from rag.search import HybridSearcher, RerankingSearcher

            s = self.settings
            if rerank != "off":
                searcher = RerankingSearcher(
                    self.conn,
                    self.searcher(mode),
                    self.reranker(),
                    s.rerank_candidates,
                    s.rerank_keep,
                    s.rerank_timeout_seconds,
                    strategy=rerank,
                    rrf_k=s.rrf_k,
                )
            elif mode == "bm25":
                searcher = self.bm25()
            elif mode == "hybrid":
                searcher = HybridSearcher(self.conn, self.dense(), self.bm25(), s.hybrid_candidates, s.rrf_k)
            else:
                searcher = self.dense()
            self._searchers[key] = searcher
        return self._searchers[key]

    def answerer(self, mode: str, rerank: str):
        key = (mode, rerank)
        if key not in self._answerers:
            from rag.answer.answerer import Answerer

            s = self.settings
            self._answerers[key] = Answerer(
                self.conn,
                self.searcher(mode, rerank),
                self.llm,
                self.prompt_counter(),
                s.ask_retrieve_k,
                s.ask_max_sources,
                s.ask_context_tokens,
                prompt_version=s.ask_prompt,
                guardrails=s.ask_guardrails,
                sanitize_sources=s.ask_sanitize_sources,
            )
        return self._answerers[key]

    def close(self) -> None:
        for searcher in self._searchers.values():
            executor = getattr(searcher, "executor", None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        self.conn.close()
