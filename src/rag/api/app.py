"""The HTTP API: the same pipeline as the CLI, loaded once and kept in memory.

    rag serve                      # http://127.0.0.1:8000, interactive docs at /docs

    GET  /health                   rag status as JSON + what the server has loaded; 503 if it can't serve
    GET  /collections              collections with document and chunk counts
    POST /search                   {"query": ..., "k": 5, "collections": [...], "mode": ..., "rerank": ...}
    POST /ask                      {"question": ..., "stream": true}  -> server-sent events:
                                     sources   what the model will read (sent before it starts)
                                     token     a piece of the answer as the model writes it (raw)
                                     done      the final answer after guardrails, citations, flags, timings
                                     error     instead of done, if generation fails midway
                                   with "stream": false, one JSON response (the done payload)

Threads. Everything that touches SQLite, the embedding model, BM25, FAISS or the reranker runs on
ONE worker thread ("pipeline"): SQLite connections belong to the thread that made them, and the
searchers keep per-call state, so requests take turns there. Generation (an HTTP call to Ollama)
runs on a second thread ("llm"), so a 7-second answer doesn't block /search. The event loop only
moves bytes, so /health answers while both are busy.

Request ids. Every response carries X-Request-ID (the caller's, if it sent a sane one); step 20's
logs will be keyed on it.
"""

import asyncio
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag.api.schemas import (
    AnswerOut,
    AskRequest,
    CollectionOut,
    SearchRequest,
    SearchResponse,
    SourcesEvent,
    answer_out,
    hit_out,
    source_out,
)
from rag.config import Settings, get_settings

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class Service:
    """What the app holds between requests: the pipeline and the two worker threads."""

    def __init__(self, settings: Settings, pipeline_factory=None, llm_probe=None):
        self.settings = settings
        self.pipeline_factory = pipeline_factory or self.default_pipeline
        self.llm_probe = llm_probe or self.default_llm_probe
        self.pipeline_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self.llm_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        self.pipeline = None
        self.load_error: str | None = None
        self.load_seconds: float | None = None
        self.started = time.time()

    def default_pipeline(self):
        from rag.pipeline import Pipeline

        if not self.settings.db_path.exists():  # never create an empty database from the server
            raise FileNotFoundError(f"no database at {self.settings.db_path} (run rag ingest first)")
        return Pipeline(self.settings)

    def default_llm_probe(self) -> str | None:
        """None if the LLM can take a request, else why not. Hosted providers aren't probed."""
        if self.settings.llm_provider != "ollama":
            return None
        from rag.status import probe_ollama

        state = probe_ollama(self.settings.ollama_base_url, self.settings.ollama_model)
        return None if state == "running" else f"LLM {self.settings.ollama_model}: {state}"

    def load(self) -> None:
        """Runs on the pipeline thread at startup: build the pipeline and warm the default
        search and ask paths, so the first request doesn't pay for loading models."""
        start = time.perf_counter()
        try:
            pipeline = self.pipeline_factory()
            s = self.settings
            pipeline.searcher(s.retrieval_mode, s.rerank)
            pipeline.answerer(s.retrieval_mode, s.ask_rerank)
            self.pipeline = pipeline
        except Exception as exc:  # the server still starts; /health says why it can't serve
            self.load_error = f"{type(exc).__name__}: {exc}"
        self.load_seconds = time.perf_counter() - start

    async def run(self, fn, *args):
        """Run fn on the pipeline thread and wait for it without blocking the event loop."""
        return await asyncio.get_running_loop().run_in_executor(self.pipeline_thread, partial(fn, *args))

    def require_pipeline(self):
        if self.pipeline is None:
            raise HTTPException(503, f"pipeline not loaded: {self.load_error or 'still starting'}")
        return self.pipeline

    def check_collections(self, names: list[str] | None) -> None:
        """Runs on the pipeline thread. Unknown names -> 404 (before any model work)."""
        if not names:
            return
        from rag.store import db

        known = {c["collection_id"] for c in db.list_collections(self.pipeline.conn)}
        unknown = sorted(set(names) - known)
        if unknown:
            raise HTTPException(404, f"no collection {', '.join(unknown)} (known: {', '.join(sorted(known))})")

    def close(self) -> None:
        self.llm_thread.shutdown(wait=False, cancel_futures=True)
        if self.pipeline is not None:
            self.pipeline_thread.submit(self.pipeline.close).result()
        self.pipeline_thread.shutdown(wait=True)


def sse(event: str, data: BaseModel | dict) -> str:
    """One server-sent event. `data` is a single line of JSON, so no escaping is needed."""
    body = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {body}\n\n"


def create_app(settings: Settings | None = None, pipeline_factory=None, llm_probe=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = Service(settings, pipeline_factory, llm_probe)
        await service.run(service.load)
        app.state.service = service
        yield
        service.close()

    app = FastAPI(
        title="RAG pipeline",
        description="Hybrid retrieval, optional reranking, grounded answers with checked citations.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        given = request.headers.get("x-request-id", "")
        request.state.request_id = given if REQUEST_ID.match(given) else uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def service_of(request: Request) -> Service:
        return request.app.state.service

    @app.get("/")
    async def root():
        return {"name": "rag", "docs": "/docs", "health": "/health"}

    @app.get("/health")
    async def health(request: Request):
        from rag.status import collect

        service = service_of(request)
        # The report without its own probes (no GROBID: not needed to serve); the LLM is probed
        # the same way /ask checks it before generating.
        report = await asyncio.to_thread(collect, settings, False)
        llm_problem = await asyncio.to_thread(service.llm_probe)
        report["llm"]["state"] = llm_problem or "running"
        problems = []
        if service.pipeline is None:
            problems.append(f"pipeline not loaded: {service.load_error or 'still starting'}")
        index = report.get("dense_index") or {}
        if index.get("state") not in (None, "current"):
            problems.append(f"dense index {index['state']}: {index.get('reason') or ''} (restart after rebuilding)")
        if llm_problem:
            problems.append(f"{llm_problem}: /ask unavailable, /search works")
        status = "unavailable" if service.pipeline is None else ("degraded" if problems else "ok")
        body = {
            "status": status,
            "problems": problems,
            "service": {
                "loaded": [] if service.pipeline is None else service.pipeline.loaded(),
                "load_seconds": service.load_seconds,
                "uptime_seconds": round(time.time() - service.started, 1),
                "defaults": {
                    "mode": settings.retrieval_mode,
                    "search_rerank": settings.rerank,
                    "ask_rerank": settings.ask_rerank,
                    "prompt": settings.ask_prompt,
                    "sanitize_sources": settings.ask_sanitize_sources,
                },
            },
            "report": report,
        }
        return JSONResponse(body, status_code=503 if status == "unavailable" else 200)

    @app.get("/collections", response_model=list[CollectionOut])
    async def collections(request: Request):
        from rag.store import db

        service = service_of(request)
        pipeline = service.require_pipeline()
        return await service.run(db.list_collections, pipeline.conn)

    @app.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request):
        service = service_of(request)
        pipeline = service.require_pipeline()
        mode = body.mode or settings.retrieval_mode
        rerank = body.rerank or settings.rerank

        def work():
            from rag.index.current import StaleIndexError

            service.check_collections(body.collections)
            try:
                searcher = pipeline.searcher(mode, rerank)
            except StaleIndexError as exc:
                raise HTTPException(503, str(exc)) from exc
            start = time.perf_counter()
            hits = searcher.search(body.query, body.k, body.collections)
            total_ms = (time.perf_counter() - start) * 1000
            return SearchResponse(
                request_id=request.state.request_id,
                query=body.query,
                mode=mode,
                rerank=rerank,
                hits=[hit_out(h, getattr(searcher, "last_ranks", None)) for h in hits],
                timings_ms={**searcher.last_timings, "total_ms": total_ms},
                fallback=getattr(searcher, "last_fallback", None),
            )

        return await service.run(work)

    @app.post("/ask", response_model=AnswerOut)
    async def ask(body: AskRequest, request: Request):
        from rag.answer.answerer import ContextWindowError
        from rag.index.current import StaleIndexError
        from rag.llm.interface import GenerationCancelled, LLMError

        service = service_of(request)
        pipeline = service.require_pipeline()
        request_id = request.state.request_id
        mode = body.mode or settings.retrieval_mode
        rerank = body.rerank or settings.ask_rerank

        llm_problem = await asyncio.to_thread(service.llm_probe)
        if llm_problem:
            raise HTTPException(503, f"{llm_problem}; /search still works")

        def prepare():
            service.check_collections(body.collections)
            try:
                answerer = pipeline.answerer(mode, rerank)
                return answerer, answerer.prepare(body.question, body.collections)
            except StaleIndexError as exc:
                raise HTTPException(503, str(exc)) from exc
            except ContextWindowError as exc:
                raise HTTPException(500, str(exc)) from exc

        answerer, prepared = await service.run(prepare)
        loop = asyncio.get_running_loop()

        if not body.stream:
            try:
                generation = await loop.run_in_executor(
                    service.llm_thread, partial(answerer.llm.generate, prepared.user_prompt, answerer.system_prompt)
                )
            except LLMError as exc:
                raise HTTPException(502, str(exc)) from exc
            answer = await service.run(answerer.finish, prepared, generation)
            return answer_out(answer, request_id, mode, rerank)

        cancelled = threading.Event()
        queue: asyncio.Queue = asyncio.Queue()

        def on_token(piece: str) -> None:  # called on the llm thread
            if cancelled.is_set():
                raise GenerationCancelled
            loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))

        def generate() -> None:  # runs on the llm thread
            try:
                result = answerer.llm.generate_stream(prepared.user_prompt, answerer.system_prompt, on_token)
                loop.call_soon_threadsafe(queue.put_nowait, ("generation", result))
            except GenerationCancelled:
                pass
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        async def events():
            try:
                yield sse(
                    "sources",
                    SourcesEvent(
                        request_id=request_id,
                        sources=[source_out(s) for s in prepared.sources],
                        timings_s=prepared.timings,
                        retrieval_fallback=prepared.retrieval_fallback,
                    ),
                )
                loop.run_in_executor(service.llm_thread, generate)
                while True:
                    kind, value = await queue.get()
                    if kind == "token":
                        yield sse("token", {"text": value})
                    elif kind == "error":
                        yield sse("error", {"request_id": request_id, "detail": f"{type(value).__name__}: {value}"})
                        return
                    else:
                        answer = await service.run(answerer.finish, prepared, value)
                        yield sse("done", answer_out(answer, request_id, mode, rerank))
                        return
            finally:  # the client left or we're done: stop the model at its next token
                cancelled.set()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run(args) -> int:
    """`rag serve`: start the API with uvicorn (one process; models load before it accepts requests)."""
    import uvicorn

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    if host not in LOOPBACK:
        print(f"refusing to listen on {host}: the API has no authentication yet (step 29); use 127.0.0.1")
        return 2
    print(f"loading the pipeline (models, indexes), then serving http://{host}:{port}  (docs: /docs)")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
    return 0
