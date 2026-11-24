"""Step 21: what the web UI needs from the API — the page itself, the Host check, stored files,
passages with highlight ranges, log records — and the guarantees around them: only files ingest
saved under data/raw can be served, and the UI never renders text as HTML."""

import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app, host_name
from rag.api.documents import stored_file
from rag.pipeline import Pipeline
from rag.store import db
from tests.test_api import StreamingFakeLLM, client_for, make_settings
from tests.test_dense import WordCounter

STATIC = Path(__file__).resolve().parents[1] / "src" / "rag" / "api" / "static"
POLICY = "Refunds are accepted within 30 days of purchase with a receipt."


@pytest.fixture
def settings(tmp_path):
    """The tiny database, plus the files ingest would have saved for two of its documents."""
    base_settings = make_settings(tmp_path)
    raw = base_settings.data_dir / "raw"
    (raw / "files").mkdir(parents=True)
    (raw / "arxiv").mkdir(parents=True)
    (raw / "files" / "file_policy.txt").write_text(POLICY)
    (raw / "arxiv" / "1706.03762.pdf").write_bytes(b"%PDF-1.4 fake attention paper")
    conn = db.open_db(base_settings.db_path)
    policy = db.get_document(conn, "file:policy")
    db.upsert_document(conn, replace(policy, source_format="txt"))
    conn.close()
    return base_settings


# --- the page and the Host check ----------------------------------------------------------------


def test_root_serves_the_ui_with_a_strict_content_security_policy(settings):
    with client_for(settings) as client:
        page = client.get("/")
        script = client.get("/static/js/main.js")
        style = client.get("/static/app.css")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert '<script type="module" src="/static/js/main.js">' in page.text
    csp = page.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert style.status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000", "localhost", "LOCALHOST:1"])
def test_loopback_hosts_are_served(settings, host):
    with client_for(settings) as client:
        assert client.get("/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "127.0.0.1.evil.example", "testserver", ""])
def test_other_hosts_are_refused(settings, host):
    """A DNS-rebinding page reaches 127.0.0.1 under its own name: the Host header gives it away."""
    with client_for(settings) as client:
        response = client.get("/collections", headers={"host": host})
    assert response.status_code == 400 and "host not allowed" in response.json()["detail"]
    assert response.headers["X-Request-ID"]


def test_default_test_client_host_is_refused(settings):
    llm = StreamingFakeLLM()
    app = create_app(settings, pipeline_factory=lambda: Pipeline(settings, llm=llm, prompt_counter=WordCounter()))
    with TestClient(app) as client:  # Host: testserver
        assert client.get("/").status_code == 400


def test_host_name():
    assert host_name("127.0.0.1:8000") == "127.0.0.1"
    assert host_name("[::1]:8000") == "::1"
    assert host_name("::1") == "::1"
    assert host_name("localhost") == "localhost"


# --- documents and their files ------------------------------------------------------------------


def test_document_info_and_files(settings):
    with client_for(settings) as client:
        info = client.get("/documents/file:policy").json()
        text = client.get("/documents/file:policy/file")
        paper = client.get("/documents/arxiv:1706.03762/file")
        paper_info = client.get("/documents/arxiv:1706.03762").json()
        no_file = client.get("/documents/arxiv:1512.03385")
    assert (info["format"], info["has_file"], info["file_url"]) == ("txt", True, "/documents/file:policy/file")
    assert text.text == POLICY and text.headers["content-type"].startswith("text/plain")
    assert text.headers["x-content-type-options"] == "nosniff"
    assert paper.headers["content-type"] == "application/pdf" and paper.content.startswith(b"%PDF")
    assert paper_info["format"] == "pdf" and paper_info["has_file"] is True
    assert no_file.json()["has_file"] is False  # in the database, but its PDF isn't on disk


@pytest.mark.parametrize(
    "path",
    [
        "/documents/../../etc/passwd/file",
        "/documents/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd/file",
        "/documents/file:..%2F..%2F..%2Fpyproject/file",
        "/documents/file:nope/file",
        "/documents/arxiv:1512.03385/file",  # known document, no stored file
    ],
)
def test_only_stored_files_of_known_documents_are_served(settings, path):
    with client_for(settings) as client:
        response = client.get(path)
    assert response.status_code == 404
    assert b"[project]" not in response.content and b"root:" not in response.content


def test_a_link_out_of_data_raw_is_not_followed(settings, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("not for the browser")
    link = settings.data_dir / "raw" / "files" / "file_policy.txt"
    link.unlink()
    link.symlink_to(outside)
    conn = db.open_db(settings.db_path)
    assert stored_file(settings.data_dir, db.get_document(conn, "file:policy")) is None
    conn.close()
    with client_for(settings) as client:
        response = client.get("/documents/file:policy/file")
    assert response.status_code == 404 and "not for the browser" not in response.text


def test_a_hostile_doc_id_stays_inside_data_raw(settings):
    """Even a doc_id written to climb out (not one ingest makes) maps to a name inside raw/files."""
    from rag.models import Document

    document = Document(doc_id="file:../../../../etc/passwd", source="file", title="x", source_format="txt")
    path = settings.data_dir / "raw" / "files" / "file_.._.._.._.._etc_passwd.txt"
    assert stored_file(settings.data_dir, document) is None  # nothing there
    path.write_text("inside")
    assert stored_file(settings.data_dir, document) == path.resolve()


# --- passages and logs --------------------------------------------------------------------------


def test_passages_give_exact_highlight_ranges(settings):
    conn = db.open_db(settings.db_path)
    chunks = db.get_chunks(conn, "file:policy")
    other = db.get_chunks(conn, "arxiv:1706.03762")[0]
    conn.close()
    with client_for(settings) as client:
        body = client.get("/passages", params={"chunk_id": [chunks[0].chunk_id]}).json()
        unknown = client.get("/passages", params={"chunk_id": ["nope"]})
        mixed = client.get("/passages", params={"chunk_id": [chunks[0].chunk_id, other.chunk_id]})
        none = client.get("/passages")
    section = body["sections"][0]
    [highlight] = section["highlights"]
    assert section["text"][highlight["start"] : highlight["end"]] == chunks[0].text
    assert body["document"]["doc_id"] == "file:policy" and highlight["page"] == 3
    assert unknown.status_code == mixed.status_code == 404
    assert none.status_code == 422


def test_log_record_of_a_request(settings):
    with client_for(settings) as client:
        client.post("/search", json={"query": "refunds"}, headers={"X-Request-ID": "ui-1"})
        record = client.get("/logs/ui-1")
        missing = client.get("/logs/nope")
    assert record.json()["route"] == "search" and record.json()["candidates"]
    assert missing.status_code == 404


def test_log_record_is_found_in_the_file_after_a_restart(settings):
    with client_for(settings) as client:
        client.post("/search", json={"query": "refunds"}, headers={"X-Request-ID": "before-restart"})
    with client_for(settings) as client:  # a new server: nothing in memory
        assert client.get("/logs/before-restart").status_code == 200


# --- the UI never renders text as HTML ----------------------------------------------------------


def test_ui_code_never_writes_html_from_strings():
    """Answers and documents are untrusted (injection-test collection): text nodes only."""
    risky = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(|new Function")
    for path in STATIC.rglob("*.js"):
        code = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("//"))
        assert not risky.search(code), path.name
    html = (STATIC / "index.html").read_text()
    assert "<script>" not in html and "onclick=" not in html  # no inline script (the CSP forbids it anyway)


def test_katex_is_vendored_and_served_from_this_origin(settings):
    """Math works offline: KaTeX's script, CSS and fonts come from /static, never a CDN."""
    html = (STATIC / "index.html").read_text()
    assert "/static/vendor/katex/katex.min.js" in html and "/static/vendor/katex/katex.min.css" in html
    assert "http://" not in html and "https://" not in html
    css = (STATIC / "vendor" / "katex" / "katex.min.css").read_text()
    for font in re.findall(r"url\((fonts/[^)]+\.woff2)\)", css):
        assert (STATIC / "vendor" / "katex" / font).is_file(), font
    with client_for(settings) as client:
        assert client.get("/static/vendor/katex/katex.min.js").status_code == 200
        assert client.get("/static/vendor/katex/fonts/KaTeX_Main-Regular.woff2").status_code == 200
