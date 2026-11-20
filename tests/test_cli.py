import subprocess
import sys

import pytest

from rag import status
from rag.chunk.chunker import Chunker
from rag.config import Settings, get_settings
from rag.index.build import EmbeddingCache, build_dense_index
from rag.main import build_parser, main
from rag.models import Document, Section
from rag.store import db
from tests.test_dense import FakeEmbedder, WordCounter

# --- argument parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "handler", "expected"),
    [
        (["status"], "rag.status:run", {"json": False}),
        (
            ["ingest", "arxiv", "ids.txt", "--collection", "papers", "--limit", "3"],
            "rag.ingest:run",
            {"source": "arxiv", "limit": 3, "collection": "papers"},
        ),
        (
            ["ingest", "files", "a.pdf", "dir/", "--collection", "hr-demo", "--reparse"],
            "rag.ingest:run",
            {"source": "files", "reparse": True, "collection": "hr-demo"},
        ),
        (["collections"], "rag.collections:run", {"action": "list"}),
        (["ingest", "tei"], "rag.ingest:run", {"source": "tei", "report": None}),
        (["parse", "x.pdf", "--doc-id", "file:x"], "rag.parse.__main__:run", {"doc_id": "file:x"}),
        (["chunk", "--show", "arxiv:1", "--limit", "5"], "rag.chunk.__main__:run", {"show": "arxiv:1", "limit": 5}),
        (["index", "build"], "rag.index.__main__:run", {"action": "build"}),
        (["search", "q", "-k", "7", "--explain"], "rag.search:run", {"query": "q", "k": 7, "collections": None}),
        (
            ["search", "q", "--collection", "papers", "--collection", "hr-demo"],
            "rag.search:run",
            {"collections": ["papers", "hr-demo"]},
        ),
        (["ask", "why?", "--collection", "papers"], "rag.ask:run", {"question": "why?", "collections": ["papers"]}),
    ],
)
def test_every_command_parses_to_its_handler(argv, handler, expected):
    args = build_parser().parse_args(argv)
    assert args.handler == handler
    assert {key: getattr(args, key) for key in expected} == expected


@pytest.mark.parametrize(
    "argv", [[], ["ingest"], ["index"], ["index", "drop"], ["search"], ["ingest", "arxiv", "ids.txt"]]
)
def test_incomplete_commands_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_every_handler_exists():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # the COMMAND subparsers
    for name, sub in commands.items():
        handler = sub.get_default("handler")
        module_name, function_name = handler.split(":")
        module = __import__(module_name, fromlist=[function_name])
        assert callable(getattr(module, function_name)), name


def test_help_and_status_do_not_load_torch_or_faiss():
    code = (
        "import sys; from rag.main import build_parser; import rag.status; build_parser(); "
        "print(sorted(m for m in ('torch', 'faiss', 'sentence_transformers', 'numpy') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_data_dir_flag_points_settings_at_another_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATA_DIR", raising=False)
    try:
        assert main(["--data-dir", str(tmp_path), "status", "--json"]) == 0
        assert get_settings().data_dir == tmp_path.resolve()
    finally:
        monkeypatch.delenv("DATA_DIR", raising=False)
        get_settings.cache_clear()
    assert f'"data_dir": "{tmp_path.resolve()}"' in capsys.readouterr().out


def test_python_dash_m_entry_points_use_the_same_parser(tmp_path, monkeypatch, capsys):
    from rag.chunk.__main__ import main as chunk_main

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rag.chunk", "--show", "arxiv:missing"])
    get_settings.cache_clear()
    try:
        assert chunk_main() == 1
    finally:
        get_settings.cache_clear()
    assert "no document arxiv:missing" in capsys.readouterr().out


# --- status -------------------------------------------------------------------


def settings_for(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path, embedding_model=FakeEmbedder.model_name, chunk_max_tokens=60, llm_provider="ollama"
    )


def add_documents(conn, chunk: bool = True, chunker_tokens: int = 60) -> None:
    chunker = Chunker(WordCounter(), chunker_tokens)
    for doc_id, text in {"arxiv:1": "Adam keeps moving averages.", "file:notes": "Residual nets train deep."}.items():
        source = doc_id.split(":")[0]
        db.upsert_document(conn, Document(doc_id=doc_id, source=source, title=doc_id, parse_status="ok"))
        sections = [Section.from_paragraphs(doc_id, 0, "1 S", 1, "1 S", [(text, 1, 1)])]
        db.replace_sections(conn, doc_id, sections)
        if chunk:
            db.replace_chunks(conn, doc_id, chunker.chunk_document(sections))


def build_index(conn, settings) -> None:
    cache = EmbeddingCache(settings.data_dir / "embeddings" / "fake.npz")
    build_dense_index(conn, FakeEmbedder(), WordCounter(), settings.data_dir / "indexes" / "dense", cache)


def test_status_on_an_empty_data_dir_suggests_ingest_and_creates_nothing(tmp_path):
    report = status.collect(settings_for(tmp_path), probe=False)
    assert report["database"] is None
    assert report["next"].startswith("rag ingest arxiv")
    assert "--collection papers" in report["next"]
    assert list(tmp_path.iterdir()) == []


def test_status_walks_through_the_next_steps(tmp_path):
    settings = settings_for(tmp_path)
    conn = db.open_db(settings.db_path)

    add_documents(conn, chunk=False)
    report = status.collect(settings, probe=False)
    assert report["database"]["unchunked_documents"] == 2
    assert report["next"] == "rag chunk"

    add_documents(conn, chunk=True)
    report = status.collect(settings, probe=False)
    assert report["database"]["documents_by_source"] == {"arxiv": 1, "file": 1}
    assert report["database"]["documents_by_collection"] == {"default": 2}
    assert report["dense_index"] == {"state": "missing"}
    assert report["next"] == "rag index build"

    build_index(conn, settings)
    report = status.collect(settings, probe=False)
    assert report["dense_index"]["state"] == "current"
    assert report["next"] == 'rag ask "<question>"'


def test_status_flags_a_stale_index_and_an_old_chunker(tmp_path):
    settings = settings_for(tmp_path)
    conn = db.open_db(settings.db_path)
    add_documents(conn)
    build_index(conn, settings)

    with conn:
        conn.execute("DELETE FROM documents WHERE doc_id = 'file:notes'")
    report = status.collect(settings, probe=False)
    assert report["dense_index"]["state"] == "stale"
    assert "rebuild" in report["dense_index"]["reason"]
    assert report["next"] == "rag index build"

    add_documents(conn, chunker_tokens=40)  # chunks from a chunker version the settings no longer use
    assert status.collect(settings, probe=False)["next"] == "rag chunk"


def test_status_suggests_starting_ollama_when_it_is_down(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    conn = db.open_db(settings.db_path)
    add_documents(conn)
    build_index(conn, settings)
    monkeypatch.setattr(status, "probe_ollama", lambda url, model: "not running")
    monkeypatch.setattr(status, "probe_grobid", lambda url: "not running")
    report = status.collect(settings)
    assert report["next"].startswith("brew services start ollama")


def test_probes_report_a_closed_port_as_not_running():
    assert status.probe_grobid("http://127.0.0.1:9") == "not running"
    assert status.probe_ollama("http://127.0.0.1:9", "qwen2.5:7b-instruct") == "not running"


def test_status_reports_an_old_schema_without_migrating_it(tmp_path):
    from tests.test_store import make_v3_database

    settings = settings_for(tmp_path)
    make_v3_database(settings.db_path)
    report = status.collect(settings, probe=False)
    assert report["schema_mismatch"] == {"database": 3, "code": db.SCHEMA_VERSION}
    assert report["next"].startswith("rag collections")
    raw = db.open_readonly(settings.db_path)
    assert db.schema_version(raw) == 3  # untouched
    assert not (tmp_path / "archive").exists()
