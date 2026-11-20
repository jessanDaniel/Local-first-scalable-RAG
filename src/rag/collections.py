"""`rag collections`: every collection with its document and chunk counts."""

from rag.config import get_settings
from rag.store import db


def run(args) -> int:
    settings = get_settings()
    if not settings.db_path.exists():
        print("no database yet (created by the first ingest)")
        return 0
    conn = db.open_db(settings.db_path)
    try:
        collections = db.list_collections(conn)
    finally:
        conn.close()
    if not collections:
        print("no collections yet")
    for c in collections:
        description = f"  {c['description']}" if c["description"] else ""
        print(f"{c['collection_id']:20} {c['n_documents']:5} documents {c['n_chunks']:6} chunks{description}")
    return 0
