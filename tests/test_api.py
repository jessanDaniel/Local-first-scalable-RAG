"""The HTTP API, offline: a tiny database, BM25 retrieval (no model downloads) and a fake LLM that
streams word by word. Real-model parity with the CLI is checked separately
(experiments/stage3/api_parity.py)."""

import json

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.chunk.chunker import Chunker
from rag.config import Settings
from rag.llm.interface import LLM, Generation, LLMError
from rag.llm.ollama import OllamaLLM
from rag.models import Document, Section
from rag.pipeline import Pipeline
from rag.store import db
from tests.test_dense import WordCounter

DOCS = {
    ("papers", "arxiv:1706.03762", "Attention Is All You Need"): (
        "We divide each dot product by sqrt dk because large dot products push softmax into regions "
        "with tiny gradients."
    ),
    ("papers", "arxiv:1512.03385", "Deep Residual Learning"): "Residual connections let very deep networks train.",
    ("hr-demo", "file:policy", "Company Policy"): "Refunds are accepted within 30 days of purchase with a receipt.",
}


class StreamingFakeLLM(LLM):
    provider = "fake"
    model = "fake-1"
    context_window = 100_000
    max_tokens = 200

    def __init__(self, text: str = "Scores are divided by sqrt dk to keep softmax gradients usable [1].", fail=False):
        self.text = text
        self.fail = fail
        self.prompts: list[tuple[str | None, str]] = []

    def generate(self, prompt, system=None):
        self.prompts.append((system, prompt))
        return Generation(self.text, self.provider, self.model, 0.01, 40, len(self.text.split()))

    def generate_stream(self, prompt, system, on_token):
        self.prompts.append((system, prompt))
        for i, word in enumerate(self.text.split(" ")):
            if self.fail and i == 2:
                raise LLMError("connection dropped")
            on_token(word if i == 0 else " " + word)
        return Generation(self.text, self.provider, self.model, 0.01, 40, len(self.text.split()))


@pytest.fixture
def settings(tmp_path):
    s = Settings(
        _env_file=None,
        data_dir=tmp_path,
        retrieval_mode="bm25",
        rerank="off",
        ask_rerank="off",
        llm_provider="ollama",
        chunk_max_tokens=60,
    )
    conn = db.open_db(s.db_path)
    chunker = Chunker(WordCounter(), 60)
    for (collection, doc_id, title), text in DOCS.items():
        db.ensure_collection(conn, collection)
        db.upsert_document(
            conn,
            Document(
                doc_id=doc_id, source=doc_id.split(":")[0], title=title, parse_status="ok", collection_id=collection
            ),
        )
        sections = [Section.from_paragraphs(doc_id, 0, "1 Body", 1, "1 Body", [(text, 3, 3)])]
        db.replace_sections(conn, doc_id, sections)
        db.replace_chunks(conn, doc_id, chunker.chunk_document(sections))
    conn.close()
    return s


def client_for(settings, llm=None, llm_probe=lambda: None, factory=None):
    llm = llm or StreamingFakeLLM()
    factory = factory or (lambda: Pipeline(settings, llm=llm, prompt_counter=WordCounter()))
    return TestClient(create_app(settings, pipeline_factory=factory, llm_probe=llm_probe))


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


# --- health, ids, collections ------------------------------------------------------------------


def test_health_reports_what_is_loaded_and_why_it_is_degraded(settings):
    with client_for(settings) as client:
        response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded"  # no dense index built in this data dir
    assert any("dense index missing" in p for p in body["problems"])
    assert "bm25" in body["service"]["loaded"] and "reranker" not in body["service"]["loaded"]
    assert body["report"]["llm"]["state"] == "running"


def test_request_id_is_echoed_or_made(settings):
    with client_for(settings) as client:
        given = client.get("/", headers={"X-Request-ID": "trace-42"})
        made = client.get("/")
        junk = client.get("/", headers={"X-Request-ID": "bad id with spaces"})
    assert given.headers["X-Request-ID"] == "trace-42"
    assert len(made.headers["X-Request-ID"]) == 16
    assert junk.headers["X-Request-ID"] != "bad id with spaces"


def test_collections(settings):
    with client_for(settings) as client:
        body = client.get("/collections").json()
    assert [(c["collection_id"], c["n_documents"]) for c in body] == [("hr-demo", 1), ("papers", 2)]


# --- search ------------------------------------------------------------------------------------


def test_search_returns_hits_with_ranks_and_timings(settings):
    with client_for(settings) as client:
        body = client.post("/search", json={"query": "why divide dot product by sqrt dk", "k": 2}).json()
    assert body["mode"] == "bm25" and body["rerank"] == "off"
    assert body["hits"][0]["doc_id"] == "arxiv:1706.03762"
    assert body["hits"][0]["page_start"] == 3 and "sqrt dk" in body["hits"][0]["text"]
    assert "search_ms" in body["timings_ms"] and "total_ms" in body["timings_ms"]


def test_search_matches_the_pipeline_called_directly(settings):
    query = "refunds within 30 days"
    with client_for(settings) as client:
        api = [h["chunk_id"] for h in client.post("/search", json={"query": query, "k": 3}).json()["hits"]]
    pipeline = Pipeline(settings, llm=StreamingFakeLLM(), prompt_counter=WordCounter())
    direct = [h.chunk.chunk_id for h in pipeline.searcher("bm25").search(query, 3)]
    assert api == direct


def test_search_collection_filter_and_errors(settings):
    with client_for(settings) as client:
        only_hr = client.post("/search", json={"query": "dot product refunds", "collections": ["hr-demo"]}).json()
        unknown = client.post("/search", json={"query": "x", "collections": ["nope"]})
        empty = client.post("/search", json={"query": ""})
        too_many = client.post("/search", json={"query": "x", "k": 500})
        bad_mode = client.post("/search", json={"query": "x", "mode": "magic"})
    assert {h["doc_id"] for h in only_hr["hits"]} == {"file:policy"}
    assert unknown.status_code == 404 and "no collection nope" in unknown.json()["detail"]
    assert empty.status_code == too_many.status_code == bad_mode.status_code == 422


# --- ask ---------------------------------------------------------------------------------------


def test_ask_streams_sources_then_tokens_then_done(settings):
    llm = StreamingFakeLLM()
    with client_for(settings, llm) as client:
        response = client.post("/ask", json={"question": "Why divide by sqrt dk?"}, headers={"X-Request-ID": "r1"})
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response.text)
    kinds = [kind for kind, _ in stream]
    assert kinds[0] == "sources" and kinds[-1] == "done" and set(kinds[1:-1]) == {"token"}
    assert stream[0][1]["sources"][0]["doc_id"] == "arxiv:1706.03762"
    streamed = "".join(data["text"] for kind, data in stream if kind == "token")
    done = stream[-1][1]
    assert streamed == done["raw_text"] == llm.text
    assert done["request_id"] == "r1" and done["citations"]["cited"] == [1]
    assert done["generation"]["model"] == "fake-1" and "total_s" in done["timings_s"]


def test_streamed_and_cli_answers_send_the_same_prompt(settings):
    question = "Why divide by sqrt dk?"
    llm = StreamingFakeLLM()
    with client_for(settings, llm) as client:
        client.post("/ask", json={"question": question})
        client.post("/ask", json={"question": question, "stream": False})
    cli = StreamingFakeLLM()
    Pipeline(settings, llm=cli, prompt_counter=WordCounter()).answerer("bm25", "off").ask(question)
    assert llm.prompts[0] == llm.prompts[1] == cli.prompts[0]


def test_ask_without_streaming_applies_guardrails(settings):
    refusal = "The sources do not say how many vacation days there are [1]."
    with client_for(settings, StreamingFakeLLM(refusal)) as client:
        body = client.post("/ask", json={"question": "How many vacation days?", "stream": False}).json()
    assert body["refusal"] is True
    assert "[1]" in body["raw_text"] and "[1]" not in body["text"]  # refusal citations stripped
    assert body["sources"]


def test_llm_down_ask_503_search_still_works(settings):
    with client_for(settings, llm_probe=lambda: "LLM qwen: not running") as client:
        ask = client.post("/ask", json={"question": "Why sqrt dk?"})
        search = client.post("/search", json={"query": "sqrt dk"})
        health = client.get("/health").json()
    assert ask.status_code == 503 and "/search still works" in ask.json()["detail"]
    assert search.status_code == 200
    assert any("/ask unavailable" in p for p in health["problems"])


def test_generation_failing_midway_sends_an_error_event(settings):
    with client_for(settings, StreamingFakeLLM(fail=True)) as client:
        stream = events(client.post("/ask", json={"question": "Why sqrt dk?"}).text)
    kinds = [kind for kind, _ in stream]
    assert kinds == ["sources", "token", "token", "error"]
    assert "connection dropped" in stream[-1][1]["detail"]


def test_pipeline_that_cannot_load_makes_health_and_search_503(settings):
    def broken():
        raise FileNotFoundError("no database")

    with client_for(settings, factory=broken) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "x"})
    assert health.status_code == 503 and health.json()["status"] == "unavailable"
    assert search.status_code == 503 and "no database" in search.json()["detail"]


# --- Ollama streaming and rag serve -------------------------------------------------------------


def test_ollama_stream_passes_pieces_on_and_keeps_counts(monkeypatch):
    lines = [
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 2, "eval_duration": 5e8},
    ]
    sent = {}

    def fake_lines(url, payload, **kwargs):
        sent.update(payload)
        yield from lines

    monkeypatch.setattr("rag.llm.ollama.post_json_lines", fake_lines)
    llm = OllamaLLM("m", "http://localhost:11434", 30, 0, 0.0, 8192, 600)
    pieces = []
    generation = llm.generate_stream("q", "sys", pieces.append)
    assert pieces == ["Hello", " world"] and generation.text == "Hello world"
    assert (generation.prompt_tokens, generation.completion_tokens, generation.write_seconds) == (12, 2, 0.5)
    assert sent["stream"] is True and sent["options"] == llm.payload("q", "sys", False)["options"]


def test_ollama_stream_without_final_line_is_an_error(monkeypatch):
    def cut_short(*args, **kwargs):
        yield {"message": {"content": "Hi"}}

    monkeypatch.setattr("rag.llm.ollama.post_json_lines", cut_short)
    with pytest.raises(LLMError, match="without a final message"):
        OllamaLLM("m", "http://x", 30, 0, 0.0, 8192, 600).generate_stream("q", None, lambda piece: None)


def test_serve_refuses_a_non_loopback_host(capsys):
    from rag.main import main

    assert main(["serve", "--host", "0.0.0.0"]) == 2
    assert "no authentication" in capsys.readouterr().out
