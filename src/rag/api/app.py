"""The HTTP API: the same pipeline as the CLI, loaded once and kept in memory.

    rag serve                      # http://127.0.0.1:8000: the web UI; interactive API docs at /docs

    GET  /                         the web UI (static files in rag/api/static, no build step)
    GET  /health                   rag status as JSON + what the server has loaded; 503 if it can't serve
    GET  /collections              collections with document and chunk counts
    GET  /documents/{doc_id}       title, format, whether the file can be opened
    GET  /documents/{doc_id}/file  the stored PDF / text file (only files ingest saved under data/raw)
    GET  /passages?chunk_id=...    the sections holding these chunks, with ranges to highlight
    GET  /logs/{request_id}        the step-20 log record of a request (the UI's trace panel)
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

Host check. Only requests whose Host header is 127.0.0.1, localhost or ::1 are served (400
otherwise). Binding to loopback isn't enough on its own: with DNS rebinding, a web page from
another site can make the browser send requests to 127.0.0.1 under its own host name; the Host
header gives that away. Origin checks and the rest of localhost hardening are step 29.

Request ids. Every response carries X-Request-ID (the caller's, if it sent a sane one). Every
/search and /ask writes one record under that id to data/logs/requests.jsonl (rag.requestlog), the
same record `rag search` / `rag ask` write: `rag logs show <id>` explains the request.
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
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
from rag.requestlog import RequestLog, Trace, stats

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
STATIC = Path(__file__).with_name("static")
# The UI page: scripts, styles and fetches from this origin only; nothing may frame it.
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def host_name(header: str) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1'; '[::1]:8000' -> '::1'; 'localhost' -> 'localhost'."""
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


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
        self.log = RequestLog.from_settings(settings)

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
        if host_name(request.headers.get("host", "")).lower() not in LOOPBACK:
            response = JSONResponse(
                {"detail": "host not allowed: open the app at http://127.0.0.1 or http://localhost"},
                status_code=400,
            )
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def service_of(request: Request) -> Service:
        return request.app.state.service

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def ui():
        return FileResponse(STATIC / "index.html", headers=UI_HEADERS)

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
            "requests": {
                "logged": service.log.written,
                "log": str(service.log.path) if service.log.path else None,
                "log_text": service.log.log_text,
                "write_errors": service.log.write_errors,
                "last_write_error": service.log.last_error,
                "recent": stats(list(service.log.recent)),
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

    def document_or_404(pipeline, doc_id: str):
        from rag.store import db

        document = db.get_document(pipeline.conn, doc_id)
        if document is None:
            raise HTTPException(404, f"no document {doc_id}")
        return document

    @app.get("/documents/{doc_id:path}/file")
    async def document_file(doc_id: str, request: Request):
        from rag.api.documents import DOWNLOAD_ONLY, MEDIA_TYPES, file_format, stored_file

        service = service_of(request)
        pipeline = service.require_pipeline()
        document = await service.run(document_or_404, pipeline, doc_id)
        path = stored_file(settings.data_dir, document)
        if path is None:
            raise HTTPException(404, f"no stored file for {doc_id}")
        fmt = file_format(document)
        disposition = f'attachment; filename="{path.name}"' if fmt in DOWNLOAD_ONLY else "inline"
        return FileResponse(
            path,
            media_type=MEDIA_TYPES[fmt],
            headers={
                "Content-Disposition": disposition,
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox",  # even if shown, nothing in it may run
            },
        )

    @app.get("/documents/{doc_id:path}")
    async def document(doc_id: str, request: Request):
        from rag.api.documents import document_info

        service = service_of(request)
        pipeline = service.require_pipeline()

        def work():
            return document_info(pipeline.conn, settings.data_dir, document_or_404(pipeline, doc_id))

        return await service.run(work)

    @app.get("/passages")
    async def passages(request: Request, chunk_id: Annotated[list[str], Query(min_length=1, max_length=20)]):
        from rag.api.documents import passage_context

        service = service_of(request)
        pipeline = service.require_pipeline()
        context = await service.run(passage_context, pipeline.conn, settings.data_dir, chunk_id)
        if context is None:
            raise HTTPException(404, "unknown chunk id, or chunks from more than one document")
        return context

    @app.get("/logs/{request_id}")
    async def log_record(request_id: str, request: Request):
        from rag.requestlog import read_records

        service = service_of(request)
        for record in reversed(service.log.recent):
            if record["request_id"] == request_id:
                return record
        if service.log.path is not None:
            records = await asyncio.to_thread(read_records, service.log.path, settings.log_backups)
            for record in reversed(records):
                if record["request_id"] == request_id:
                    return record
        raise HTTPException(404, f"no logged request {request_id}")

    @app.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request):
        service = service_of(request)
        mode = body.mode or settings.retrieval_mode
        rerank = body.rerank or settings.rerank
        trace = Trace("api", "search", body.query, mode, rerank, body.collections, request.state.request_id, k=body.k)

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
            trace.searched(searcher, hits, total_ms)
            return SearchResponse(
                request_id=request.state.request_id,
                query=body.query,
                mode=mode,
                rerank=rerank,
                hits=[hit_out(h, getattr(searcher, "last_ranks", None)) for h in hits],
                timings_ms={**searcher.last_timings, "total_ms": total_ms},
                fallback=getattr(searcher, "last_fallback", None),
            )

        try:
            pipeline = service.require_pipeline()
            return await service.run(work)
        except HTTPException as exc:
            trace.failed(str(exc.detail), http_status=exc.status_code)
            raise
        finally:
            service.log.write(trace.record())

    @app.post("/ask", response_model=AnswerOut)
    async def ask(body: AskRequest, request: Request):
        from rag.answer.answerer import ContextWindowError
        from rag.index.current import StaleIndexError
        from rag.llm.interface import GenerationCancelled, LLMError

        service = service_of(request)
        request_id = request.state.request_id
        mode = body.mode or settings.retrieval_mode
        rerank = body.rerank or settings.ask_rerank
        trace = Trace("api", "ask", body.question, mode, rerank, body.collections, request_id)

        def prepare():
            service.check_collections(body.collections)
            try:
                answerer = pipeline.answerer(mode, rerank)
                prepared = answerer.prepare(body.question, body.collections)
            except StaleIndexError as exc:
                raise HTTPException(503, str(exc)) from exc
            except ContextWindowError as exc:
                raise HTTPException(500, str(exc)) from exc
            trace.prepared(answerer, prepared)
            return answerer, prepared

        loop = asyncio.get_running_loop()
        try:
            pipeline = service.require_pipeline()
            check_start = time.perf_counter()
            llm_problem = await asyncio.to_thread(service.llm_probe)
            trace.time("llm_check_ms", (time.perf_counter() - check_start) * 1000)
            if llm_problem:
                raise HTTPException(503, f"{llm_problem}; /search still works")
            answerer, prepared = await service.run(prepare)
            if not body.stream:
                try:
                    generation = await loop.run_in_executor(
                        service.llm_thread,
                        partial(answerer.llm.generate, prepared.user_prompt, answerer.system_prompt),
                    )
                except LLMError as exc:
                    raise HTTPException(502, str(exc)) from exc
                trace.generated(generation)
                answer = await service.run(answerer.finish, prepared, generation)
                trace.answered(answer)
                service.log.write(trace.record())
                return answer_out(answer, request_id, mode, rerank)
        except HTTPException as exc:
            trace.failed(str(exc.detail), http_status=exc.status_code)
            service.log.write(trace.record())
            raise

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
            logged = False
            streamed = 0
            first_token_ms = None
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
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - trace.started) * 1000
                        streamed += 1
                        yield sse("token", {"text": value})
                    elif kind == "error":
                        detail = f"{type(value).__name__}: {value}"
                        trace.failed(detail)
                        trace.data["streamed_pieces"] = streamed
                        service.log.write(trace.record())
                        logged = True
                        yield sse("error", {"request_id": request_id, "detail": detail})
                        return
                    else:
                        trace.generated(value, first_token_ms)
                        answer = await service.run(answerer.finish, prepared, value)
                        trace.answered(answer)
                        service.log.write(trace.record())
                        logged = True
                        yield sse("done", answer_out(answer, request_id, mode, rerank))
                        return
            finally:  # the client left or we're done: stop the model at its next token
                cancelled.set()
                if not logged:
                    trace.failed(f"client disconnected after {streamed} streamed pieces", status="cancelled")
                    trace.data["streamed_pieces"] = streamed
                    service.log.write(trace.record())

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
    print(f"loading the pipeline (models, indexes), then serving the app at http://{host}:{port}  (API docs: /docs)")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
    return 0
