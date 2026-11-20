"""`rag status`: what is stored, whether the index matches it, which services are up, what to run next.

Light imports only (no torch, no FAISS): it reads SQLite, the index's JSON files,
and probes GROBID and the LLM with a 1-second timeout. It never writes: a
missing database is reported, not created. The same report is what the
service's /health endpoint will return (Stage 3).
"""

import json
from urllib.parse import urlparse

import requests

from rag.chunk.chunker import CHUNKER_VERSION
from rag.config import Settings, get_settings
from rag.index.current import current_build_dir, database_state, read_chunk_ids, read_manifest, stale_reason
from rag.main import DEFAULT_SEED
from rag.store import db

PROBE_TIMEOUT_SECONDS = 1.0
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def probe_get(url: str) -> requests.Response:
    """GET with a short timeout. Local services skip any HTTP(S)_PROXY from the environment: a proxy
    answers for localhost with its own error (e.g. 503), which would look like a sick service."""
    with requests.Session() as session:
        session.trust_env = urlparse(url).hostname not in LOCAL_HOSTS
        return session.get(url, timeout=PROBE_TIMEOUT_SECONDS)


def probe_grobid(base_url: str) -> str:
    try:
        response = probe_get(f"{base_url}/api/isalive")
    except requests.RequestException:
        return "not running"
    return "running" if response.ok and response.text.strip() == "true" else f"unhealthy (HTTP {response.status_code})"


def probe_ollama(base_url: str, model: str) -> str:
    try:
        response = probe_get(f"{base_url}/api/tags")
        response.raise_for_status()
        names = {m.get("name") for m in response.json().get("models", [])}
    except (requests.RequestException, ValueError):
        return "not running"
    if model in names or f"{model}:latest" in names:
        return "running"
    return f"running, but model {model} is not pulled (ollama pull {model})"


def dense_index_report(settings: Settings, conn) -> dict:
    build_dir = current_build_dir(settings.data_dir / "indexes" / "dense")
    if build_dir is None:
        return {"state": "missing"}
    manifest = read_manifest(build_dir)
    db_chunk_ids, fingerprint = database_state(conn)
    reason = stale_reason(manifest, read_chunk_ids(build_dir), settings.embedding_model, db_chunk_ids, fingerprint)
    return {
        "state": "stale" if reason else "current",
        "reason": reason,
        "build_id": manifest.build_id,
        "index_type": manifest.index_type,
        "n_chunks": manifest.n_chunks,
        "embedding_model": manifest.embedding_model,
        "chunker_version": manifest.chunker_version,
    }


def collect(settings: Settings, probe: bool = True) -> dict:
    chunker = f"{CHUNKER_VERSION}-{settings.chunk_max_tokens}"
    report: dict = {"data_dir": str(settings.data_dir), "database": None, "chunker_setting": chunker}
    if settings.db_path.exists():
        conn = db.open_readonly(settings.db_path)
        try:
            version = db.schema_version(conn)
            if version == db.SCHEMA_VERSION:
                report["database"] = db.stats(conn)
                report["dense_index"] = dense_index_report(settings, conn)
            else:
                report["schema_mismatch"] = {"database": version, "code": db.SCHEMA_VERSION}
        finally:
            conn.close()

    if settings.llm_provider == "ollama":
        llm = {"provider": "ollama", "model": settings.ollama_model, "url": settings.ollama_base_url}
        if probe:
            llm["state"] = probe_ollama(settings.ollama_base_url, settings.ollama_model)
    else:
        llm = {"provider": settings.llm_provider, "model": settings.groq_model, "state": "hosted (not probed)"}
    report["llm"] = llm
    report["grobid"] = {"url": settings.grobid_url}
    if probe:
        report["grobid"]["state"] = probe_grobid(settings.grobid_url)
    report["next"] = next_step(report)
    return report


def next_step(report: dict) -> str:
    mismatch = report.get("schema_mismatch")
    if mismatch and mismatch["database"] < mismatch["code"]:
        return "rag collections   (any command that opens the database migrates it; a backup is kept)"
    if mismatch:
        return "update the code: the database was made by a newer version"
    stats = report["database"]
    if not stats or not sum(stats["documents_by_status"].values()):
        grobid_down = report["grobid"].get("state", "running") != "running"  # not probed: assume fine
        start = "start GROBID (see AGENTS.md), then: " if grobid_down else ""
        alternative = "(or: rag ingest files <paths> --collection NAME)"
        return f"{start}rag ingest arxiv {DEFAULT_SEED} --collection papers   {alternative}"
    if stats["unchunked_documents"] or set(stats["chunks_by_chunker"]) - {report["chunker_setting"]}:
        return "rag chunk"
    if report["dense_index"]["state"] != "current":
        return "rag index build"
    if report["llm"].get("state", "running") not in ("running", "hosted (not probed)"):
        return 'brew services start ollama, then: rag ask "<question>"'
    return 'rag ask "<question>"'


def print_report(report: dict) -> None:
    rows = [("data", report["data_dir"])]
    stats = report["database"]
    mismatch = report.get("schema_mismatch")
    if mismatch:
        rows.append(("database", f"rag.db, schema v{mismatch['database']}; code expects v{mismatch['code']}"))
    elif stats is None:
        rows.append(("database", "none yet (created by the first ingest)"))
    else:
        by_status = ", ".join(f"{k} {v}" for k, v in sorted(stats["documents_by_status"].items()))
        by_source = ", ".join(f"{k} {v}" for k, v in sorted(stats["documents_by_source"].items()))
        n_docs = sum(stats["documents_by_status"].values())
        rows.append(("database", f"rag.db, schema v{stats['schema_version']}"))
        rows.append(("documents", f"{n_docs}" + (f" ({by_status}; sources: {by_source})" if n_docs else "")))
        by_collection = ", ".join(f"{k} {v}" for k, v in sorted(stats["documents_by_collection"].items()))
        rows.append(("collections", by_collection or "none"))
        rows.append(("sections", str(stats["sections"])))
        versions = ", ".join(stats["chunks_by_chunker"]) or "none"
        chunks = f"{stats['chunks']} (chunker {versions}; settings {report['chunker_setting']})"
        if stats["unchunked_documents"]:
            chunks += f", {stats['unchunked_documents']} parsed documents not chunked"
        rows.append(("chunks", chunks))
        index = report["dense_index"]
        if index["state"] == "missing":
            rows.append(("index", "none built"))
        else:
            line = f"{index['build_id']} {index['index_type']}, {index['n_chunks']} chunks, {index['embedding_model']}"
            rows.append(("index", f"{line} - {index['state']}"))
            if index["reason"]:
                rows.append(("", index["reason"]))
    grobid = report["grobid"]
    rows.append(("GROBID", f"{grobid['url']} - {grobid.get('state', 'not probed')} (needed only to ingest PDFs)"))
    llm = report["llm"]
    where = f" at {llm['url']}" if "url" in llm else ""
    rows.append(("LLM", f"{llm['provider']} {llm['model']}{where} - {llm.get('state', 'not probed')}"))
    rows.append(("next", report["next"]))
    for label, value in rows:
        print(f"{label:11} {value}")


def run(args) -> int:
    report = collect(get_settings())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0
