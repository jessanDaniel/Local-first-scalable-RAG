"""`rag collections`: list collections, or delete one.

rag collections                    # every collection with its document and chunk counts, and folders
rag collections delete NAME        # what would be removed (nothing is deleted)
rag collections delete NAME --yes  # remove it: documents, chunks, stored copies, uploads; the
                                   # index is rebuilt. Files in its folders are never touched.
"""

from rag.config import get_settings
from rag.store import db


def run(args) -> int:
    settings = get_settings()
    if not settings.db_path.exists():
        print("no database yet (created by the first ingest)")
        return 0
    conn = db.open_db(settings.db_path)
    try:
        if getattr(args, "action", "list") == "delete":
            return delete(settings, conn, args.name, args.yes)
        collections = db.list_collections(conn)
        folders = db.source_folders(conn)
    finally:
        conn.close()
    if not collections:
        print("no collections yet")
    for c in collections:
        description = f"  {c['description']}" if c["description"] else ""
        print(f"{c['collection_id']:20} {c['n_documents']:5} documents {c['n_chunks']:6} chunks{description}")
        for f in folders:
            if f["collection_id"] == c["collection_id"]:
                print(f"{'':20} folder {f['path']}  (synced {f['last_synced'] or 'never'})")
    return 0


def delete(settings, conn, name: str | None, yes: bool) -> int:
    from rag.sync import delete_collection, rebuild_index

    known = {c["collection_id"]: c for c in db.list_collections(conn)}
    if not name or name not in known:
        print(f"error: no collection {name} (known: {', '.join(sorted(known)) or 'none'})")
        return 1
    c = known[name]
    if not yes:
        print(
            f"would remove {name}: {c['n_documents']} documents, {c['n_chunks']} chunks, the app's stored copies "
            f"and uploads. Files in its folders are not touched. Run again with --yes to do it."
        )
        return 1
    result = delete_collection(settings, conn, name)
    build_id, _ = rebuild_index(settings, conn)
    print(f"removed {name} ({result['documents_removed']} documents); index {build_id or 'none (no chunks left)'}")
    return 0
