"""Steps 24 + 30 through the API: collections, folders, uploads, deletes, jobs, the Origin check,
and a running server answering from new documents without a restart."""

import time

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app, port_of, same_origin
from rag.pipeline import Pipeline
from tests.test_api import StreamingFakeLLM, make_settings
from tests.test_dense import FakeEmbedder, WordCounter

BODY = " ".join(["Visitors sign in at the front desk and wear a badge at all times."] * 30)


@pytest.fixture
def settings(tmp_path):
    s = make_settings(tmp_path / "app")  # the app's data folder; documents live elsewhere in tmp_path
    return s.model_copy(update={"embedding_model": FakeEmbedder.model_name, "upload_max_mb": 0.05})


def app_client(settings):
    llm = StreamingFakeLLM()
    app = create_app(
        settings,
        pipeline_factory=lambda: Pipeline(settings, llm=llm, prompt_counter=WordCounter()),
        llm_probe=lambda: None,
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    return TestClient(app, base_url="http://127.0.0.1")


def wait(client, job_id, timeout=20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {job['status']}")


def search_ids(client, query, collection):
    """Documents in the hits, in order, each once (a document can have several matching chunks)."""
    hits = client.post("/search", json={"query": query, "collections": [collection]}).json()["hits"]
    return list(dict.fromkeys(h["doc_id"] for h in hits))


# --- the Origin check ---------------------------------------------------------------------------


def test_cross_site_writes_are_refused_and_the_apps_own_pages_are_not(settings):
    with app_client(settings) as client:
        evil = client.post("/collections", json={"name": "x"}, headers={"origin": "https://evil.example"})
        sandboxed = client.post("/collections", json={"name": "y"}, headers={"origin": "null"})
        other_port = client.post("/collections", json={"name": "z"}, headers={"origin": "http://127.0.0.1:9999"})
        own = client.post("/collections", json={"name": "ok"}, headers={"origin": "http://127.0.0.1"})
        localhost = client.post("/collections", json={"name": "ok2"}, headers={"origin": "http://localhost"})
        no_origin = client.post("/collections", json={"name": "ok3"})  # curl, the CLI
        read = client.get("/collections", headers={"origin": "https://evil.example"})  # CORS stops the read
    assert evil.status_code == sandboxed.status_code == other_port.status_code == 403
    assert "cross-site request refused" in evil.json()["detail"]
    assert own.status_code == localhost.status_code == no_origin.status_code == 201
    assert read.status_code == 200


def test_same_origin_and_port_of():
    assert same_origin("http://127.0.0.1:8000", "127.0.0.1:8000") and same_origin(
        "http://localhost:8000", "127.0.0.1:8000"
    )
    assert same_origin("http://[::1]:8000", "[::1]:8000") and same_origin("http://127.0.0.1", "127.0.0.1")
    assert not same_origin("http://127.0.0.1:8001", "127.0.0.1:8000") and not same_origin(
        "http://evil.test", "evil.test"
    )
    assert (port_of("127.0.0.1:8000"), port_of("[::1]:8765"), port_of("localhost")) == ("8000", "8765", "80")


# --- collections and folders --------------------------------------------------------------------


def test_create_collection_rules(settings):
    with app_client(settings) as client:
        made = client.post("/collections", json={"name": "site-docs", "description": "Office rules"})
        again = client.post("/collections", json={"name": "site-docs"})
        bad = client.post("/collections", json={"name": "Bad Name!"})
        listed = [c["collection_id"] for c in client.get("/collections").json()]
    assert made.status_code == 201 and again.status_code == 409 and bad.status_code == 422
    assert "site-docs" in listed


def test_folder_added_synced_and_searchable_without_a_restart(settings, tmp_path):
    folder = tmp_path / "office"
    (folder / "rules").mkdir(parents=True)
    (folder / "rules" / "visitors.md").write_text(f"# Visitors\n\n## Sign in\n\n{BODY}\n")
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        assert search_ids(client, "visitors badge front desk", "office") == []
        added = client.post("/collections/office/folders", json={"path": str(folder)})
        assert added.status_code == 202
        job = wait(client, added.json()["job"]["id"])
        assert job["status"] == "done" and job["result"]["added"] == ["file:rules-visitors"]
        assert any(e["kind"] == "swapped" for e in job["events"])
        assert search_ids(client, "visitors badge front desk", "office") == ["file:rules-visitors"]

        detail = client.get("/collections/office").json()
        [doc] = detail["documents"]
        assert (doc["origin"], doc["file"], detail["folders"][0]["exists"]) == ("folder", "visitors.md", True)

        # a file added later, then a rescan
        (folder / "parking.md").write_text(f"# Parking\n\n## Bays\n\n{BODY} Bays are numbered.\n")
        job = wait(client, client.post("/collections/office/sync").json()["job"]["id"])
        assert job["result"]["added"] == ["file:parking"] and job["result"]["unchanged"] == 1

        # the folder removed: its documents go
        removed = client.delete("/collections/office/folders", params={"path": str(folder.resolve())})
        assert removed.json()["documents_removed"] == 2
        wait(client, removed.json()["job"]["id"])
        assert search_ids(client, "visitors badge front desk", "office") == []


def test_folder_rules(settings, tmp_path):
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        relative = client.post("/collections/office/folders", json={"path": "docs"})
        missing = client.post("/collections/office/folders", json={"path": str(tmp_path / "nope")})
        own_data = client.post("/collections/office/folders", json={"path": str(settings.data_dir)})
        parent = client.post("/collections/office/folders", json={"path": str(settings.data_dir.parent)})
        (tmp_path / "docs").mkdir()
        unknown = client.post("/collections/nope/folders", json={"path": str(tmp_path / "docs")})
    assert relative.status_code == missing.status_code == own_data.status_code == parent.status_code == 422
    assert unknown.status_code == 404


# --- uploads and deletes -----------------------------------------------------------------------


def test_upload_then_sync_then_delete(settings):
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        put = client.put("/collections/office/files/fire drill.md", content=f"# Fire\n\n## Drill\n\n{BODY}".encode())
        assert put.status_code == 201 and put.json()["bytes"] > 0
        job = wait(client, client.post("/collections/office/sync").json()["job"]["id"])
        assert job["result"]["added"] == ["file:upload-fire-drill"]
        assert search_ids(client, "visitors badge front desk", "office") == ["file:upload-fire-drill"]
        assert client.get("/collections/office").json()["documents"][0]["origin"] == "upload"

        deleted = client.delete("/documents/file:upload-fire-drill")
        assert deleted.status_code == 202
        wait(client, deleted.json()["job"]["id"])
        assert search_ids(client, "visitors badge front desk", "office") == []
    assert not (settings.data_dir / "raw" / "uploads" / "office" / "fire drill.md").exists()


@pytest.mark.parametrize("name", [".hidden.md", "~$draft.docx", "run.exe", "notes", "a\\b.md", "x" * 151 + ".md"])
def test_refused_upload_names(settings, name):
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        response = client.put(f"/collections/office/files/{name}", content=b"hello")
    assert response.status_code in (404, 422)  # 404: the router never matched (a slash in the name)


def test_upload_limits(settings):
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        too_big = client.put("/collections/office/files/big.txt", content=b"x" * 60_000)  # limit 0.05 MB
        empty = client.put("/collections/office/files/empty.txt", content=b"")
        unknown = client.put("/collections/nope/files/a.txt", content=b"hello")
    assert too_big.status_code == 413 and empty.status_code == 422 and unknown.status_code == 404
    leftovers = list((settings.data_dir / "raw" / "uploads" / "office").iterdir())
    assert leftovers == []  # no partial file left behind


def test_a_folder_document_cannot_be_deleted_on_its_own(settings, tmp_path):
    folder = tmp_path / "office"
    folder.mkdir()
    (folder / "visitors.md").write_text(f"# Visitors\n\n## Sign in\n\n{BODY}\n")
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        wait(client, client.post("/collections/office/folders", json={"path": str(folder)}).json()["job"]["id"])
        response = client.delete("/documents/file:visitors")
    assert response.status_code == 409 and "comes from the folder" in response.json()["detail"]


# --- jobs ---------------------------------------------------------------------------------------


def test_jobs_list_cancel_and_unknown(settings):
    with app_client(settings) as client:
        client.post("/collections", json={"name": "office"})
        job = client.post("/collections/office/sync").json()["job"]
        cancel = client.post(f"/jobs/{job['id']}/cancel")
        finished = wait(client, job["id"])
        listed = client.get("/jobs").json()
        missing = client.get("/jobs/nope")
    assert cancel.json()["cancel_requested"] is True
    assert finished["status"] in ("done", "cancelled")  # it may finish before the cancel lands
    assert listed[0]["id"] == job["id"] and missing.status_code == 404


def test_startup_rescans_registered_folders(settings, tmp_path):
    from rag.store import db

    folder = tmp_path / "office"
    folder.mkdir()
    (folder / "visitors.md").write_text(f"# Visitors\n\n## Sign in\n\n{BODY}\n")
    conn = db.open_db(settings.db_path)
    db.ensure_collection(conn, "office")
    db.add_source_folder(conn, "office", str(folder.resolve()))
    conn.close()
    with app_client(settings) as client:
        [job] = client.get("/jobs").json()
        assert job["reason"] == "startup rescan"
        assert wait(client, job["id"])["result"]["added"] == ["file:visitors"]
