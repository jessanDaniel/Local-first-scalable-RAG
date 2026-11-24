"""What the API accepts and returns. FastAPI validates requests against these (a bad request is a
422 with the field that failed) and lists them in /docs.

Converters from the pipeline's own dataclasses (Hit, Answer) live here too, so the routes stay
thin and the JSON shape is defined in one place.
"""

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["hybrid", "dense", "bm25"]
Rerank = Literal["off", "fused", "replace"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    collections: list[str] | None = Field(default=None, description="only these collections (default: all)")
    mode: Mode | None = Field(default=None, description="default: RETRIEVAL_MODE setting")
    rerank: Rerank | None = Field(default=None, description="default: RERANK setting")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    collections: list[str] | None = None
    mode: Mode | None = Field(default=None, description="default: RETRIEVAL_MODE setting")
    rerank: Rerank | None = Field(default=None, description="default: ASK_RERANK setting")
    stream: bool = Field(default=True, description="server-sent events (true) or one JSON response (false)")


class HitOut(BaseModel):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: str
    page_start: int | None
    page_end: int | None
    chunk_type: str
    text: str
    source_uri: str | None
    ranks: dict[str, int | None] = Field(description="rank in each retriever list: which one brought it in")


class SearchResponse(BaseModel):
    request_id: str
    query: str
    mode: Mode
    rerank: Rerank
    hits: list[HitOut]
    timings_ms: dict[str, float]
    fallback: str | None = Field(description="why the reranker was skipped, if it was")


class SourceOut(BaseModel):
    n: int
    doc_id: str
    title: str
    section_path: str
    pages: str
    page_start: int | None
    page_end: int | None
    link: str | None
    chunk_ids: list[str]
    best_rank: int
    text: str


class FlagOut(BaseModel):
    kind: str
    detail: str
    sentence: str | None = None


class CitationsOut(BaseModel):
    cited: list[int]
    invalid: list[int]
    n_sentences: int
    n_cited_sentences: int


class GenerationOut(BaseModel):
    provider: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    prompt_seconds: float | None
    write_seconds: float | None
    load_seconds: float | None


class AnswerOut(BaseModel):
    """The `done` event of a streamed answer, and the whole response when stream=false."""

    request_id: str
    question: str
    mode: Mode
    rerank: Rerank
    text: str = Field(description="after guardrails: what to show")
    raw_text: str = Field(description="what the model wrote (the streamed tokens)")
    refusal: bool
    citations: CitationsOut
    sources: list[SourceOut]
    flags: list[FlagOut]
    warnings: list[str]
    retrieval_fallback: str | None
    prompt_version: str
    prompt_tokens: int
    generation: GenerationOut
    timings_s: dict[str, float]


class SourcesEvent(BaseModel):
    """First event of a streamed answer: what the model is about to read."""

    request_id: str
    sources: list[SourceOut]
    timings_s: dict[str, float]
    retrieval_fallback: str | None


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64, description="lower case letters, digits and hyphens")
    description: str = Field(default="", max_length=500)


class FolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096, description="a folder on this machine")


class CollectionOut(BaseModel):
    collection_id: str
    description: str | None
    n_documents: int
    n_chunks: int
    # The overview on the Documents tab (step 22).
    created_at: str | None = None
    n_suspect: int = 0  # parsed, but a quality gate raised a doubt
    n_failed: int = 0
    n_folders: int = 0
    n_missing_folders: int = 0  # registered, but not a folder on this machine any more
    last_synced: str | None = None  # the most recent folder sync (UTC)


# --- converters ------------------------------------------------------------------------------


def hit_out(hit, ranks: dict | None) -> HitOut:
    c = hit.chunk
    return HitOut(
        rank=hit.rank,
        score=hit.score,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        doc_title=hit.doc_title,
        section_path=hit.section_path,
        page_start=c.page_start,
        page_end=c.page_end,
        chunk_type=c.chunk_type,
        text=c.text,
        source_uri=hit.source_uri,
        ranks=(ranks or {}).get(c.chunk_id, {}),
    )


def source_out(source) -> SourceOut:
    return SourceOut(
        n=source.n,
        doc_id=source.doc_id,
        title=source.title,
        section_path=source.section_path,
        pages=source.pages,
        page_start=source.page_start,
        page_end=source.page_end,
        link=source.link,
        chunk_ids=source.chunk_ids,
        best_rank=source.best_rank,
        text=source.text,
    )


def answer_out(answer, request_id: str, mode: str, rerank: str) -> AnswerOut:
    g = answer.generation
    c = answer.citations
    return AnswerOut(
        request_id=request_id,
        question=answer.question,
        mode=mode,
        rerank=rerank,
        text=answer.text,
        raw_text=answer.raw_text,
        refusal=answer.refusal,
        citations=CitationsOut(
            cited=c.cited, invalid=c.invalid, n_sentences=c.n_sentences, n_cited_sentences=c.n_cited_sentences
        ),
        sources=[source_out(s) for s in answer.sources],
        flags=[FlagOut(kind=f.kind, detail=f.detail, sentence=f.sentence) for f in answer.flags],
        warnings=answer.warnings,
        retrieval_fallback=answer.retrieval_fallback,
        prompt_version=answer.prompt_version,
        prompt_tokens=answer.prompt_tokens,
        generation=GenerationOut(
            provider=g.provider,
            model=g.model,
            prompt_tokens=g.prompt_tokens,
            completion_tokens=g.completion_tokens,
            prompt_seconds=g.prompt_seconds,
            write_seconds=g.write_seconds,
            load_seconds=g.load_seconds,
        ),
        timings_s=answer.timings,
    )
