"""Step 24: rag sync (only what changed is redone) and swapping new indexes into a running pipeline."""

import argparse
import threading
from pathlib import Path

import pytest

from rag import sync as sync_module
from rag.config import Settings
from rag.pipeline import Pipeline
from rag.store import db
from rag.sync import prune_builds, sync_collection
from tests.test_dense import FakeEmbedder, WordCounter

BODY = " ".join(["Employees book travel through the portal and keep receipts for thirty days."] * 30)


@pytest.fixture
def setup(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", embedding_model=FakeEmbedder.model_name)
    folder = tmp_path / "docs"
    (folder / "hr").mkdir(parents=True)
    (folder / "travel.md").write_text(f"# Travel\n\n## Booking\n\n{BODY}\n\n## Hotels\n\nHotels near the office.\n")
    (folder / "hr" / "leave.md").write_text(f"# Leave\n\n## Annual\n\n{BODY} Annual leave is 25 days.\n")
    conn = db.open_db(settings.db_path)
    db.ensure_collection(conn, "team")
    db.add_source_folder(conn, "team", str(folder.resolve()))
    conn.close()
    return settings, folder


def run_sync(settings, **kwargs):
    events = []
    report = sync_collection(
        settings, "team", on_event=events.append, embedder=FakeEmbedder(), counter=WordCounter(), **kwargs
    )
    return report, events


def documents(settings) -> set[str]:
    conn = db.open_db(settings.db_path)
    ids = {d.doc_id for d in db.list_documents(conn, collection_id="team")}
    conn.close()
    return ids


def test_first_sync_adds_everything_and_a_second_changes_nothing(setup):
    settings, _ = setup
    first, _ = run_sync(settings)
    assert sorted(first.added) == ["file:hr-leave", "file:travel"] and first.build_id and first.n_chunks > 0
    second, _ = run_sync(settings)
    assert sorted(second.unchanged) == ["file:hr-leave", "file:travel"]
    assert (second.added, second.updated, second.removed, second.build_id) == ([], [], [], None)


def test_a_changed_file_is_reparsed_and_reindexed(setup):
    settings, folder = setup
    run_sync(settings)
    (folder / "hr" / "leave.md").write_text(f"# Leave\n\n## Annual\n\n{BODY} Annual leave is 30 days now.\n")
    report, _ = run_sync(settings)
    assert report.updated == ["file:hr-leave"] and report.unchanged == ["file:travel"]
    assert report.build_id and report.n_embedded > 0  # only the changed chunks were embedded
    conn = db.open_db(settings.db_path)
    assert any("30 days now" in c.text for c in db.get_chunks(conn, "file:hr-leave"))
    conn.close()


def test_a_deleted_file_is_removed_with_its_chunks_and_stored_copy(setup):
    settings, folder = setup
    run_sync(settings)
    conn = db.open_db(settings.db_path)
    stored = Path(db.get_document(conn, "file:travel").source_uri)
    conn.close()
    assert stored.exists()
    (folder / "travel.md").unlink()
    report, events = run_sync(settings)
    assert report.removed == ["file:travel"] and documents(settings) == {"file:hr-leave"}
    assert not stored.exists() and any(e.kind == "removed" for e in events)


def test_a_missing_folder_keeps_its_documents(setup, tmp_path):
    settings, folder = setup
    run_sync(settings)
    folder.rename(tmp_path / "unplugged")
    report, _ = run_sync(settings)
    assert report.missing_folders == [str(folder.resolve())] and report.removed == []
    assert documents(settings) == {"file:hr-leave", "file:travel"}


def test_documents_added_another_way_are_never_removed_by_a_sync(setup, tmp_path):
    from rag.ingest import Ingestor

    settings, _ = setup
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_text(f"# Other\n\n## Part\n\n{BODY}\n")
    ingestor = Ingestor(settings, collection="team")
    ingestor.ingest_files([elsewhere])
    ingestor.close()
    run_sync(settings)
    elsewhere.unlink()
    report, _ = run_sync(settings)
    assert report.removed == [] and "file:elsewhere" in documents(settings)


def test_uploads_are_part_of_the_collection(setup):
    settings, _ = setup
    uploads = settings.data_dir / "raw" / "uploads" / "team"
    uploads.mkdir(parents=True)
    (uploads / "notes.md").write_text(f"# Notes\n\n## A\n\n{BODY}\n")
    report, _ = run_sync(settings)
    assert "file:upload-notes" in report.added
    (uploads / "notes.md").unlink()
    assert run_sync(settings)[0].removed == ["file:upload-notes"]


def test_a_cancelled_sync_keeps_what_it_finished(setup):
    settings, _ = setup
    cancel = threading.Event()
    seen = []

    def stop_after_first(event):
        seen.append(event)
        if event.kind == "added":
            cancel.set()

    report = sync_collection(
        settings, "team", on_event=stop_after_first, cancel=cancel, embedder=FakeEmbedder(), counter=WordCounter()
    )
    assert report.cancelled and len(report.added) == 1 and len(documents(settings)) == 1
    assert report.build_id  # what was finished was chunked and indexed: database and index agree
    conn = db.open_db(settings.db_path)
    assert db.count_chunks(conn) > 0
    conn.close()
    report, _ = run_sync(settings)  # the next sync goes on from there
    assert len(report.added) == 1 and len(report.unchanged) == 1


def test_remove_folder_deletes_its_documents_and_rebuilds(setup, monkeypatch, capsys):
    settings, folder = setup
    run_sync(settings)
    monkeypatch.setattr(sync_module, "get_settings", lambda: settings)
    real = sync_module.sync_collection
    monkeypatch.setattr(
        sync_module,
        "sync_collection",
        lambda s, c, **kw: real(s, c, embedder=FakeEmbedder(), counter=WordCounter(), **kw),
    )
    args = argparse.Namespace(collection="team", add_folder=None, remove_folder=folder)
    assert sync_module.run(args) == 0
    assert documents(settings) == set() and "and its 2 documents" in capsys.readouterr().out


def test_prune_keeps_the_newest_builds_and_current(tmp_path):
    for name in ["dense-1", "dense-2", "dense-3", "dense-4", "dense-5"]:
        (tmp_path / name).mkdir()
    (tmp_path / "CURRENT").write_text("dense-1")  # pointing at an old one (a rollback)
    removed = prune_builds(tmp_path, keep=3)
    assert sorted(removed) == ["dense-2"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["dense-1", "dense-3", "dense-4", "dense-5"]


# --- a running pipeline picks up a sync without a restart -------------------------------------


def test_reload_swaps_in_the_new_indexes(setup):
    settings, folder = setup
    run_sync(settings)
    pipeline = Pipeline(settings, prompt_counter=WordCounter())
    pipeline._embedder = FakeEmbedder()
    hybrid = pipeline.searcher("hybrid")
    assert not any("helicopter" in h.chunk.text for h in hybrid.search("helicopter pad rules", 3, ["team"]))

    (folder / "site.md").write_text(f"# Site\n\n## Roof\n\n{BODY} The helicopter pad rules are strict.\n")
    run_sync(settings)
    info = pipeline.reload()
    assert info["dense_build"] and info["bm25_chunks"]
    hits = pipeline.searcher("hybrid").search("helicopter pad rules", 3, ["team"])
    assert "helicopter" in hits[0].chunk.text
    pipeline.close()


def test_search_skips_chunks_deleted_before_the_swap(setup):
    settings, folder = setup
    run_sync(settings)
    pipeline = Pipeline(settings, prompt_counter=WordCounter())
    pipeline._embedder = FakeEmbedder()
    searcher = pipeline.searcher("hybrid")
    (folder / "travel.md").unlink()
    run_sync(settings)  # deletes file:travel; the pipeline still holds the old indexes
    hits = searcher.search("travel portal receipts", 5, ["team"])
    assert all(h.chunk.doc_id != "file:travel" for h in hits)  # left out, no error
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    pipeline.close()


def test_a_file_whose_id_another_collection_has_gets_the_collection_prefix(setup, tmp_path):
    settings, _ = setup
    other = tmp_path / "other"
    other.mkdir()
    (other / "travel.md").write_text(f"# Travel elsewhere\n\n## Part\n\n{BODY} Different.\n")
    conn = db.open_db(settings.db_path)
    db.ensure_collection(conn, "other")
    db.add_source_folder(conn, "other", str(other.resolve()))
    conn.close()
    sync_collection(settings, "other", embedder=FakeEmbedder(), counter=WordCounter())  # takes file:travel
    report, _ = run_sync(settings)
    assert sorted(report.added) == ["file:hr-leave", "file:team-travel"] and report.failed == []
    assert run_sync(settings)[0].removed == []  # stable on the next sync


def test_a_file_that_fails_keeps_its_old_document(setup, monkeypatch):
    from rag.ingest import Ingestor

    settings, folder = setup
    run_sync(settings)
    (folder / "travel.md").write_text(f"# Travel\n\n## Booking\n\n{BODY} changed\n")
    real = Ingestor.ingest_files

    def flaky(self, items):
        if any(getattr(i, "key", "") == "travel" for i in items):
            raise OSError("disk hiccup")
        return real(self, items)

    monkeypatch.setattr(Ingestor, "ingest_files", flaky)
    report, _ = run_sync(settings)
    assert report.failed and report.removed == [] and "file:travel" in documents(settings)
