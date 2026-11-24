"""Background jobs for the API: syncing a collection (rag.sync), one at a time.

Parsing and embedding already use the CPU and GPU fully, and SQLite allows one writer, so jobs run
on a single worker thread and queue behind each other. A job's progress is the list of sync events
(added, updated, removed, failed, skipped, indexed ...) that the UI polls. When a sync changed
something, the server's pipeline swaps the new indexes in (Pipeline.reload, on the pipeline thread)
before the job is marked done, so the next request already sees the new documents.

The job thread has its own database connection (the sync's) and its own embedding model for new
chunks; the pipeline thread keeps serving /search and /ask the whole time (SQLite WAL lets it read
while the sync writes).
"""

import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

MAX_JOBS_KEPT = 50


class Lazy:
    """Stands in for an object that is expensive to build (the embedding model, a tokenizer): built
    by `factory` on first attribute use and kept. A sync with nothing new never touches it, so a
    rescan of unchanged folders takes milliseconds instead of loading a model."""

    def __init__(self, factory):
        self._factory = factory
        self._target = None

    def __getattr__(self, name):
        if self._target is None:
            self._target = self._factory()
        return getattr(self._target, name)


@dataclass
class Job:
    id: str
    kind: str  # sync
    collection: str
    reason: str  # what started it: "folder added", "upload", "rescan", "startup", ...
    status: str = "queued"  # queued | running | done | failed | cancelled
    events: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def view(self, events_from: int = 0) -> dict:
        """As JSON (by hand: asdict would deep-copy the cancel Event, which holds a lock)."""
        fields = ("id", "kind", "collection", "reason", "status", "created", "started", "finished", "result", "error")
        out = {name: getattr(self, name) for name in fields}
        out["n_events"] = len(self.events)
        out["events"] = self.events[events_from:]
        return out


class JobManager:
    def __init__(self, settings, swap_indexes, embedder_factory=None, counter_factory=None):
        """swap_indexes(): called after a sync that changed something (the server reloads its
        pipeline there). The factories build the job thread's embedder / token counter once."""
        self.settings = settings
        self.swap_indexes = swap_indexes
        self.embedder_factory = embedder_factory
        self.counter_factory = counter_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jobs")
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.lock = threading.Lock()
        self._embedder = Lazy(embedder_factory) if embedder_factory else None
        self._counter = Lazy(counter_factory) if counter_factory else None

    def submit_sync(self, collection: str, reason: str, rebuild: bool = False) -> Job:
        job = Job(uuid.uuid4().hex[:12], "sync", collection, reason)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > MAX_JOBS_KEPT:
                self.jobs.popitem(last=False)
        self.executor.submit(self._run_sync, job, rebuild)
        return job

    def submit_rebuild(self, collection: str, reason: str) -> Job:
        """Rebuild the index from what the database holds (after a collection was deleted) and swap
        it in. Queued behind any running sync, like every job."""
        job = Job(uuid.uuid4().hex[:12], "rebuild", collection, reason)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > MAX_JOBS_KEPT:
                self.jobs.popitem(last=False)
        self.executor.submit(self._run_rebuild, job)
        return job

    def _run_rebuild(self, job: Job) -> None:
        from rag.store import db
        from rag.sync import rebuild_index

        job.status, job.started = "running", time.time()
        conn = None
        try:
            conn = db.open_db(self.settings.db_path)
            build_id, n_embedded = rebuild_index(
                self.settings,
                conn,
                self._embedder,
                self._counter,
                lambda kind, name, detail="": job.events.append({"kind": kind, "name": name, "detail": detail}),
            )
            swapped = self.swap_indexes()
            job.events.append({"kind": "swapped", "name": "indexes", "detail": f"{swapped}"})
            job.result = {"build_id": build_id, "n_embedded": n_embedded, "swap": swapped}
            job.status = "done"
        except Exception as exc:  # a job must never take the server down
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if conn is not None:
                conn.close()
            job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def recent(self) -> list[Job]:
        return list(reversed(self.jobs.values()))

    def active(self, collection: str | None = None) -> list[Job]:
        return [
            j for j in self.jobs.values() if j.status in ("queued", "running") and collection in (None, j.collection)
        ]

    def _run_sync(self, job: Job, rebuild: bool) -> None:
        from rag.sync import sync_collection

        if job.cancel.is_set():
            job.status, job.finished = "cancelled", time.time()
            return
        job.status, job.started = "running", time.time()
        try:
            report = sync_collection(
                self.settings,
                job.collection,
                on_event=lambda e: job.events.append({"kind": e.kind, "name": e.name, "detail": e.detail}),
                cancel=job.cancel,
                embedder=self._embedder,
                counter=self._counter,
                rebuild=rebuild,
            )
            swapped = None
            if report.changed or rebuild:
                swapped = self.swap_indexes()
                job.events.append({"kind": "swapped", "name": "indexes", "detail": f"{swapped}"})
            job.result = {
                "added": report.added,
                "updated": report.updated,
                "unchanged": len(report.unchanged),
                "removed": report.removed,
                "failed": report.failed,
                "skipped": report.skipped,
                "missing_folders": report.missing_folders,
                "build_id": report.build_id,
                "n_embedded": report.n_embedded,
                "seconds": {k: round(v, 2) for k, v in report.seconds.items()},
                "swap": swapped,
                "cancelled": report.cancelled,
            }
            job.status = "cancelled" if report.cancelled else "done"
        except Exception as exc:  # a job must never take the server down
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()

    def close(self) -> None:
        for job in self.jobs.values():
            job.cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
