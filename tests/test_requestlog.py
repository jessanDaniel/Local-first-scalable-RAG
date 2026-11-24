"""Step 20: per-request logs. The API and the CLI write the same record; a request can be explained
from it (stages, candidates, sources, tokens, answer, checks); errors and cancelled streams are
logged too; LOG_TEXT=false keeps no text; files rotate."""

import argparse
import json
import time

from rag import ask as ask_cli
from rag import requestlog
from rag import search as search_cli
from rag.config import Settings
from rag.pipeline import Pipeline
from rag.requestlog import RequestLog, Trace, read_records, redact, show, stats
from tests import test_api
from tests.test_api import StreamingFakeLLM, client_for, events
from tests.test_dense import WordCounter

settings = test_api.settings  # the same tiny database fixture

QUESTION = "Why divide by sqrt dk?"


def log_path(s: Settings):
    return s.data_dir / "logs" / "requests.jsonl"


def logged(s: Settings) -> list[dict]:
    return read_records(log_path(s))


def with_settings(s: Settings, **changes) -> Settings:
    return s.model_copy(update=changes)


# --- the API writes one record per request ------------------------------------------------------


def test_search_record_explains_the_request(settings):
    with client_for(settings) as client:
        response = client.post("/search", json={"query": "sqrt dk softmax", "k": 2}, headers={"X-Request-ID": "s1"})
    [record] = logged(settings)
    assert record["request_id"] == response.headers["X-Request-ID"] == "s1"
    assert (record["source"], record["route"], record["status"]) == ("api", "search", "ok")
    assert record["settings"]["mode"] == "bm25" and record["settings"]["k"] == 2
    assert record["query"] == "sqrt dk softmax"
    assert [c["chunk_id"] for c in record["candidates"]] == [h["chunk_id"] for h in response.json()["hits"]]
    t = record["timings_ms"]
    assert {"search_ms", "bm25_search_ms", "fetch_ms", "total_ms", "other_ms"} <= set(t)
    assert abs(t["search_ms"] + t["other_ms"] - t["total_ms"]) < 0.05  # the named stages add up


def test_streamed_ask_record_has_every_stage_tokens_and_the_answer(settings):
    llm = StreamingFakeLLM()
    with client_for(settings, llm) as client:
        response = client.post("/ask", json={"question": QUESTION})
    done = events(response.text)[-1][1]
    [record] = logged(settings)
    assert record["request_id"] == done["request_id"] and record["status"] == "ok"
    t = record["timings_ms"]
    for stage in ("llm_check_ms", "retrieve_ms", "assemble_ms", "generate_ms", "finish_ms", "request_first_token_ms"):
        assert stage in t, stage
    top = t["llm_check_ms"] + t["retrieve_ms"] + t["assemble_ms"] + t["generate_ms"] + t["finish_ms"]
    assert abs(top + t["other_ms"] - t["total_ms"]) < 0.05
    assert record["llm"]["model"] == "fake-1" and record["llm"]["prompt_tokens_counted"] > 0
    assert record["candidates"][0]["doc_id"] == "arxiv:1706.03762" and record["candidates"][0]["source"] == 1
    assert record["sources"][0]["chunk_ids"] == done["sources"][0]["chunk_ids"]
    assert record["answer"]["text"] == done["text"] and record["answer"]["cited"] == [1]
    assert record["settings"]["prompt"] == settings.ask_prompt


def test_cli_and_api_write_the_same_fields(settings, monkeypatch, capsys):
    llm = StreamingFakeLLM()
    with client_for(settings, llm) as client:
        client.post("/ask", json={"question": QUESTION, "stream": False})
        client.post("/search", json={"query": "sqrt dk", "k": 3})
    api_ask, api_search = logged(settings)

    for module in (ask_cli, search_cli):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(ask_cli, "Pipeline", lambda s: Pipeline(s, llm=llm, prompt_counter=WordCounter()))
    args = argparse.Namespace(question=QUESTION, collections=None, mode=None, rerank=None, explain=False)
    assert ask_cli.run(args) == 0
    assert (
        search_cli.run(
            argparse.Namespace(query="sqrt dk", k=3, collections=None, mode=None, rerank=None, explain=False)
        )
        == 0
    )
    cli_ask, cli_search = logged(settings)[2:]
    assert f"rag logs show {cli_ask['request_id']}" in capsys.readouterr().out

    assert cli_ask["source"] == "cli" and api_ask["source"] == "api"
    assert set(cli_ask) == set(api_ask)
    assert set(cli_ask["timings_ms"]) == set(api_ask["timings_ms"]) - {"llm_check_ms"}
    assert cli_ask["candidates"] == api_ask["candidates"] and cli_ask["sources"] == api_ask["sources"]
    assert set(cli_search) == set(api_search) and cli_search["candidates"] == api_search["candidates"]


# --- failures are logged too --------------------------------------------------------------------


def test_errors_are_logged_with_their_status(settings):
    with client_for(settings, StreamingFakeLLM(fail=True)) as client:
        client.post("/search", json={"query": "x", "collections": ["nope"]})
        client.post("/ask", json={"question": QUESTION})
    unknown, midway = logged(settings)
    assert unknown["status"] == "error" and unknown["http_status"] == 404 and "no collection nope" in unknown["error"]
    assert midway["status"] == "error" and "connection dropped" in midway["error"]
    assert midway["streamed_pieces"] == 2 and midway["candidates"]  # retrieval is still in the record


def test_llm_down_is_logged(settings):
    with client_for(settings, llm_probe=lambda: "LLM qwen: not running") as client:
        client.post("/ask", json={"question": QUESTION})
    [record] = logged(settings)
    assert record["http_status"] == 503 and "not running" in record["error"]
    assert "llm_check_ms" in record["timings_ms"]


class SlowFakeLLM(StreamingFakeLLM):
    """Writes one word every 50 ms, so a client can leave while it is still writing."""

    def generate_stream(self, prompt, system, on_token):
        def slow(piece):
            time.sleep(0.05)
            on_token(piece)

        return super().generate_stream(prompt, system, slow)


def test_a_client_leaving_mid_stream_is_logged_as_cancelled(settings):
    llm = SlowFakeLLM(" ".join(f"w{i}" for i in range(60)))
    with client_for(settings, llm) as client:
        with client.stream("POST", "/ask", json={"question": QUESTION}) as response:
            for line in response.iter_lines():
                if line.startswith("event: token"):
                    break
    [record] = logged(settings)
    assert record["status"] == "cancelled" and "disconnected" in record["error"]
    assert record["streamed_pieces"] < 60 and record["candidates"]


# --- settings, privacy, rotation ----------------------------------------------------------------


def test_log_text_off_keeps_no_question_or_answer_text(settings):
    quiet = with_settings(settings, log_text=False)
    answer = "Scores are divided by sqrt dk to keep softmax gradients usable [1]."
    with client_for(quiet, StreamingFakeLLM(answer)) as client:
        client.post("/ask", json={"question": "Why divide by the magic number?"})
        health = client.get("/health").json()
    raw = log_path(quiet).read_text()
    assert "magic number" not in raw and "gradients usable" not in raw
    [record] = logged(quiet)
    assert record["text_logged"] is False and record["answer"]["cited"] == [1]
    assert health["requests"]["log_text"] is False


def test_log_requests_off_writes_no_file(settings):
    off = with_settings(settings, log_requests=False)
    with client_for(off) as client:
        client.post("/search", json={"query": "sqrt dk"})
        health = client.get("/health").json()
    assert not log_path(off).exists()
    assert health["requests"]["recent"]["search"]["requests"] == 1  # /health still counts


def test_rotation_keeps_backups_and_reads_back_in_order(tmp_path):
    log = RequestLog(tmp_path / "requests.jsonl", max_bytes=600, backups=2)
    for i in range(12):
        trace = Trace("cli", "search", f"query {i}", "bm25", "off", None, request_id=f"r{i:02d}")
        log.write(trace.record())
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["requests.1.jsonl", "requests.2.jsonl", "requests.jsonl"]
    ids = [r["request_id"] for r in read_records(tmp_path / "requests.jsonl", backups=2)]
    assert ids == sorted(ids) and ids[-1] == "r11" and len(ids) < 12  # oldest ones deleted


def test_a_failing_write_does_not_fail_the_request(tmp_path):
    blocker = tmp_path / "logs"
    blocker.write_text("a file where the folder should be")
    log = RequestLog(blocker / "requests.jsonl")
    log.write(Trace("cli", "search", "q", "bm25", "off", None).record())
    assert log.write_errors == 1 and log.last_error and len(log.recent) == 1


def test_redact_keeps_kinds_and_numbers():
    record = {
        "query": "secret question",
        "answer": {"text": "a", "raw_text": "b", "flags": [{"kind": "number_not_in_source", "detail": "42 x"}]},
    }
    out = redact(record)
    assert out["query"] is None and out["answer"]["flags"] == [
        {"kind": "number_not_in_source", "detail": None, "sentence": None}
    ]
    assert record["query"] == "secret question"  # the caller's record is untouched


# --- reading the logs ---------------------------------------------------------------------------


def test_show_and_stats(settings):
    refusal = "The sources do not say how many vacation days there are [1]."
    with client_for(settings, StreamingFakeLLM(refusal)) as client:
        client.post("/ask", json={"question": "How many vacation days?"}, headers={"X-Request-ID": "vac-1"})
        for _ in range(3):
            client.post("/search", json={"query": "sqrt dk"})
    records = logged(settings)
    text = show(records[0])
    assert "request vac-1" in text and "REFUSAL" in text and "guardrails edited it" in text
    assert "retrieve" in text and "candidates" in text and "-> [1]" in text
    summary = stats(records)
    assert summary["ask"]["refusals"] == 1 and summary["search"]["requests"] == 3
    assert summary["search"]["stages_ms"]["total_ms"]["n"] == 3


def test_rag_logs_command(settings, monkeypatch, capsys):
    with client_for(settings) as client:
        client.post("/search", json={"query": "sqrt dk"}, headers={"X-Request-ID": "abc123"})
    monkeypatch.setattr("rag.config.get_settings", lambda: settings)
    assert requestlog.run(argparse.Namespace(action="show", request_id="abc", json=False)) == 0
    assert "request abc123" in capsys.readouterr().out
    assert requestlog.run(argparse.Namespace(action="show", request_id="zzz", json=False)) == 1
    capsys.readouterr()
    assert requestlog.run(argparse.Namespace(action="stats", last=50, route=None, source=None, json=True)) == 0
    assert json.loads(capsys.readouterr().out)["search"]["ok"] == 1


def test_ollama_stream_reports_time_to_first_token(monkeypatch):
    from rag.llm.ollama import OllamaLLM

    def fake_lines(url, payload, **kwargs):
        yield {"message": {"content": "Hi"}, "done": False}
        yield {"message": {"content": ""}, "done": True, "prompt_eval_count": 3, "eval_count": 1}

    monkeypatch.setattr("rag.llm.ollama.post_json_lines", fake_lines)
    generation = OllamaLLM("m", "http://x", 30, 0, 0.0, 8192, 600).generate_stream("q", None, lambda piece: None)
    assert generation.first_token_seconds is not None and generation.first_token_seconds <= generation.elapsed_seconds
