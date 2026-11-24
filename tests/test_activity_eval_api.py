"""Step 22: the read-only views behind the Activity and Evaluation tabs and the collections overview.
The API gives the same numbers as the CLI (`rag logs stats`, `rag eval compare`), run ids can't name
a file outside the runs directory, and nothing here writes anything."""

import argparse
import json

import pytest

from rag import requestlog
from rag.eval import gate as gate_module
from rag.eval.answers import save_answer_run
from rag.eval.runner import compare_runs, run_diff, save_run
from rag.requestlog import RequestLog, Trace, for_stats, iter_newest, read_records, stats
from rag.store import db
from tests import test_api
from tests.test_answer_eval import record as answer_record
from tests.test_api import StreamingFakeLLM, client_for
from tests.test_eval import fake_run

settings = test_api.settings  # the same tiny database fixture

REFUSAL = "The sources do not say how many vacation days there are [1]."


def runs_dir(s):
    return s.data_dir / "eval" / "runs"


def save(s, run, created):
    run["created"] = created
    run["run_id"] = f"{run['set']}-{created}" + (f"-{run['label']}" if run["label"] else "")
    save_run(run, runs_dir(s))
    return run


@pytest.fixture
def with_runs(settings):
    """Three papers runs and one hr-demo run, saved the way `rag eval run` saves them."""
    a = save(
        settings,
        fake_run("papers", {"q1": 1, "q2": 3, "q3": None}, label="base", retriever="hybrid"),
        "20261101T000000Z",
    )
    b = save(
        settings,
        fake_run("papers", {"q1": 1, "q2": 7, "q3": 2}, label="fused", retriever="hybrid", rerank="fused"),
        "20261102T000000Z",
    )
    old = save(
        settings, fake_run("papers", {"q1": 2, "q2": 2, "q3": 2}, label="old", retriever="rerank"), "20261001T000000Z"
    )
    hr = save(settings, fake_run("hr-demo", {"q1": 1}), "20261103T000000Z")
    return {"a": a, "b": b, "old": old, "hr": hr}


# --- recent requests ----------------------------------------------------------------------------


def test_logs_list_is_newest_first_with_filters_and_pages(settings):
    with client_for(settings, StreamingFakeLLM(REFUSAL)) as client:
        client.post(
            "/ask", json={"question": "How many vacation days?", "stream": False}, headers={"X-Request-ID": "ask-1"}
        )
        for i in range(3):
            client.post("/search", json={"query": "sqrt dk"}, headers={"X-Request-ID": f"s-{i}"})
        everything = client.get("/logs").json()
        asks = client.get("/logs", params={"route": "ask"}).json()
        page1 = client.get("/logs", params={"route": "search", "limit": 2}).json()
        page2 = client.get("/logs", params={"route": "search", "limit": 2, "offset": 2}).json()
        flagged = client.get("/logs", params={"flagged": True}).json()
        cli_only = client.get("/logs", params={"source": "cli"}).json()
        bad = client.get("/logs", params={"route": "drop"})
    assert [r["request_id"] for r in everything["records"]] == ["s-2", "s-1", "s-0", "ask-1"]
    ask = asks["records"][0]
    assert ask["query"] == "How many vacation days?" and ask["answer"]["refusal"] is True
    assert ask["route"] == "ask" and ask["source"] == "api" and ask["total_ms"] > 0
    assert [r["request_id"] for r in page1["records"]] == ["s-2", "s-1"] and page1["more"] is True
    assert [r["request_id"] for r in page2["records"]] == ["s-0"] and page2["more"] is False
    # The refusal had its citation stripped and no flag: nothing is flagged.
    assert flagged["records"] == [] and cli_only["records"] == []
    assert bad.status_code == 422


def test_logs_list_without_a_log_file_uses_the_requests_in_memory(settings):
    off = settings.model_copy(update={"log_requests": False})
    with client_for(off) as client:
        client.post("/search", json={"query": "sqrt dk"}, headers={"X-Request-ID": "mem-1"})
        body = client.get("/logs").json()
        stats_body = client.get("/logs/stats").json()
    assert [r["request_id"] for r in body["records"]] == ["mem-1"] and body["log"] is None
    assert stats_body["routes"]["search"]["requests"] == 1


def test_log_text_off_shows_no_question(settings):
    quiet = settings.model_copy(update={"log_text": False})
    with client_for(quiet) as client:
        client.post("/search", json={"query": "a private question"})
        row = client.get("/logs").json()["records"][0]
    assert row["query"] is None and row["text_logged"] is False


def test_iter_newest_reads_rotated_files_in_reverse(tmp_path):
    log = RequestLog(tmp_path / "requests.jsonl", max_bytes=600, backups=2)
    for i in range(12):
        log.write(Trace("cli", "search", f"query {i}", "bm25", "off", None, request_id=f"r{i:02d}").record())
    newest = [r["request_id"] for r in iter_newest(tmp_path / "requests.jsonl", backups=2)]
    oldest_first = [r["request_id"] for r in read_records(tmp_path / "requests.jsonl", backups=2)]
    assert newest == list(reversed(oldest_first)) and newest[0] == "r11"


def test_iter_newest_skips_a_line_cut_short(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text('{"request_id": "a"}\n{"request_id": "b"}\n{"request_id": "c', encoding="utf-8")
    assert [r["request_id"] for r in iter_newest(path)] == ["b", "a"]


# --- stats: the same numbers as `rag logs stats` ------------------------------------------------


def test_log_stats_match_the_cli(settings, monkeypatch, capsys):
    with client_for(settings, StreamingFakeLLM(REFUSAL)) as client:
        client.post("/ask", json={"question": "How many vacation days?", "stream": False})
        for _ in range(4):
            client.post("/search", json={"query": "sqrt dk"})
        api_all = client.get("/logs/stats", params={"last": 0}).json()
        api_last3 = client.get("/logs/stats", params={"last": 3}).json()
        api_ask = client.get("/logs/stats", params={"route": "ask"}).json()
    monkeypatch.setattr("rag.config.get_settings", lambda: settings)

    def cli(**args) -> dict:
        capsys.readouterr()
        namespace = argparse.Namespace(
            action="stats", json=True, **{"last": 200, "route": None, "source": None, **args}
        )
        assert requestlog.run(namespace) == 0
        return json.loads(capsys.readouterr().out)

    assert api_all["routes"] == cli(last=1000) and api_all["n_records"] == 5 and api_all["last"] is None
    assert api_last3["routes"] == cli(last=3) and api_last3["n_records"] == 3
    assert api_ask["routes"] == cli(route="ask") and set(api_ask["routes"]) == {"ask"}
    assert api_all["oldest"] <= api_all["newest"]
    assert set(cli(source="cli")) == set()  # every request here came through the API
    assert api_all["routes"] == stats(read_records(settings.data_dir / "logs" / "requests.jsonl"))


def test_stats_need_only_the_slim_records(settings):
    """/logs/stats keeps for_stats(record), not the record: same numbers, a fraction of the memory."""
    with client_for(settings, StreamingFakeLLM(REFUSAL)) as client:
        client.post("/ask", json={"question": "How many vacation days?", "stream": False})
        client.post("/ask", json={"question": "How many vacation days?", "stream": True})
        client.post("/search", json={"query": "sqrt dk"})
    records = read_records(settings.data_dir / "logs" / "requests.jsonl")
    assert stats([for_stats(r) for r in records]) == stats(records)
    assert "candidates" not in for_stats(records[0]) and "query" not in for_stats(records[0])


# --- evaluation runs ----------------------------------------------------------------------------


def test_runs_list_newest_first_and_by_set(settings, with_runs):
    with client_for(settings) as client:
        everything = client.get("/eval/runs").json()
        papers = client.get("/eval/runs", params={"set": "papers"}).json()
    assert [r["run_id"] for r in everything][:2] == [with_runs["hr"]["run_id"], with_runs["b"]["run_id"]]
    assert [r["label"] for r in papers] == ["fused", "base", "old"]
    fused, _, old = papers
    assert (fused["retriever"], fused["rerank"], fused["hit@1"], fused["n"]) == ("hybrid", "fused", 1, 3)
    assert (old["retriever"], old["rerank"]) == ("hybrid", "replace")  # a run from before step 15b
    assert fused["collections"] == ["c"]


def test_run_detail_has_every_question(settings, with_runs):
    with client_for(settings) as client:
        detail = client.get(f"/eval/runs/{with_runs['a']['run_id']}").json()
    assert {q["id"]: q["first_rank"] for q in detail["questions"]} == {"q1": 1, "q2": 3, "q3": None}
    assert detail["summary"]["overall"]["hit@5"] == 2 and detail["config"]["index_type"] == "flat"


def test_compare_is_run_diff_and_matches_the_cli_text(settings, with_runs):
    a, b = with_runs["a"], with_runs["b"]
    with client_for(settings) as client:
        body = client.get("/eval/compare", params={"a": a["run_id"], "b": b["run_id"]}).json()
    expected = run_diff(a, b)
    for flip in body["flips"]:
        assert flip.pop("question") is None  # eval/sets/papers.jsonl has no q1..q3: no wording
    assert body == json.loads(json.dumps(expected))
    assert [(f["id"], f["a"], f["b"], f["crosses"]) for f in body["flips"]] == [
        ("q3", None, 2, [5, 10]),
        ("q2", 3, 7, [5]),
    ]
    assert {c["key"] for c in body["config_changes"]} == {"rerank"}
    text = "\n".join(compare_runs(a, b))
    assert "q3" in text and ">20 -> 2" in text and "crosses top-5" in text


def test_compare_errors(settings, with_runs):
    with client_for(settings) as client:
        other_set = client.get("/eval/compare", params={"a": with_runs["a"]["run_id"], "b": with_runs["hr"]["run_id"]})
        unknown = client.get("/eval/compare", params={"a": with_runs["a"]["run_id"], "b": "papers-nope"})
        missing = client.get("/eval/compare", params={"a": with_runs["a"]["run_id"]})
    assert other_set.status_code == 422 and "different sets" in other_set.json()["detail"]
    assert unknown.status_code == 404 and "papers-nope" in unknown.json()["detail"]
    assert missing.status_code == 422


@pytest.mark.parametrize("run_id", ["..", "...json", "..%2Fbaseline", "%2E%2E", "%2Fetc%2Fpasswd", "papers:latest"])
def test_a_run_id_never_names_a_file_outside_the_runs_directory(settings, with_runs, run_id):
    """A run id is looked up in the directory listing, never joined into a path."""
    (settings.data_dir / "eval" / "secret.json").write_text(json.dumps(with_runs["a"]))
    with client_for(settings) as client:
        response = client.get(f"/eval/runs/{run_id}")
        compared = client.get("/eval/compare", params={"a": "../secret", "b": with_runs["a"]["run_id"]})
    assert response.status_code == 404
    assert compared.status_code == 404


def test_no_runs_yet_is_an_empty_list(settings):
    with client_for(settings) as client:
        assert client.get("/eval/runs").json() == []
        assert client.get("/eval/answers").json() == []


def test_a_broken_run_file_is_skipped(settings, with_runs):
    (runs_dir(settings) / "papers-broken.json").write_text("{not json")
    with client_for(settings) as client:
        runs = client.get("/eval/runs").json()
    assert len(runs) == 4


# --- answer runs and the gate -------------------------------------------------------------------


def test_answer_runs_list_their_summary(settings):
    records = [
        answer_record("a", outcomes=("correct?", "correct?")),
        answer_record("n", answerable=False, outcomes=("refused_ok", "refused_ok"), qtype="unanswerable"),
    ]
    config = {
        "llm": "ollama test",
        "prompt": "v3",
        "guardrails": True,
        "retriever": "hybrid",
        "rerank": "fused",
        "runs": 2,
    }
    run = save_answer_run("papers", config, records, settings.data_dir / "eval" / "answers", "guard")
    with client_for(settings) as client:
        listed = client.get("/eval/answers").json()
    assert [r["run_id"] for r in listed] == [run["run_id"]]
    item = listed[0]
    assert (item["prompt"], item["guardrails"], item["rerank"], item["runs"]) == ("v3", True, "fused", 2)
    assert item["summary"]["unanswerable"] == {"refused": 2, "of": 2, "refused_with_citations": 0}
    assert "attempts" not in json.dumps(item["summary"]["answerable"])  # no per-answer text is sent


def test_gate_summary(settings, tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(gate_module, "BASELINE_PATH", path)
    with client_for(settings) as client:
        assert client.get("/eval/gate").status_code == 404
        retrieval = {
            "papers": {
                "n": 3,
                "metrics": {"hit@1": 1, "hit@5": 2, "hit@10": 2, "mrr": 0.5},
                "ranks": {"q1": 1, "q2": 7, "q3": None},
            }
        }
        first = gate_module.write_baseline(
            retrieval, {"r|q1|1": [], "r|q2|1": ["entity"]}, {"retriever": "hybrid"}, "first", "abc", None, path
        )
        gate_module.write_baseline(retrieval, None, {"retriever": "hybrid"}, "second", "def", first, path)
        body = client.get("/eval/gate").json()
    assert body["reason"] == "second" and body["commit"] == "def"
    assert body["answers_replayed"] == 2 and body["answers_flagged"] == 1  # flags kept from the first update
    assert body["sets"]["papers"]["misses_top5"] == [{"id": "q2", "rank": 7}, {"id": "q3", "rank": None}]
    assert [h["reason"] for h in body["history"]] == ["second", "first"]  # newest first
    assert "ranks" not in body["sets"]["papers"]


# --- collections overview -----------------------------------------------------------------------


def test_collections_carry_the_overview_fields(settings, tmp_path):
    conn = db.open_db(settings.db_path)
    there = tmp_path / "policies"
    there.mkdir()
    db.add_source_folder(conn, "hr-demo", str(there))
    db.add_source_folder(conn, "hr-demo", str(tmp_path / "gone"))
    db.mark_folder_synced(conn, "hr-demo", str(there))
    conn.execute("UPDATE documents SET parse_status = 'suspect' WHERE collection_id = 'papers'")
    conn.commit()
    conn.close()
    with client_for(settings) as client:
        rows = {c["collection_id"]: c for c in client.get("/collections").json()}
    hr, papers = rows["hr-demo"], rows["papers"]
    assert (hr["n_folders"], hr["n_missing_folders"]) == (2, 1) and hr["last_synced"]
    assert papers["n_suspect"] == papers["n_documents"] == 2 and papers["n_folders"] == 0
    assert papers["last_synced"] is None and papers["n_failed"] == 0


# --- the page -----------------------------------------------------------------------------------


def test_the_page_has_the_activity_and_evaluation_tabs(settings):
    with client_for(settings) as client:
        page = client.get("/").text
        scripts = [client.get(f"/static/js/{name}.js") for name in ("activity", "evaluation", "overview")]
    for tab in ('id="tab-activity"', 'id="tab-eval"', 'id="view-activity"', 'id="view-eval"', 'id="overview"'):
        assert tab in page
    assert all(s.status_code == 200 for s in scripts)


def test_static_files_are_revalidated_so_an_update_never_runs_old_scripts(settings):
    """Without Cache-Control the browser may reuse a stale module after an update (seen in the step-22
    browser check: the old activity.js kept running against the new page)."""
    with client_for(settings) as client:
        script = client.get("/static/js/activity.js")
        again = client.get("/static/js/activity.js", headers={"If-None-Match": script.headers["etag"]})
        api = client.get("/collections")
    assert script.headers["cache-control"] == "no-cache"
    assert again.status_code == 304  # unchanged: revalidation costs a header, not the file
    assert "cache-control" not in api.headers
