"""Step 21b: what does giving up GROBID cost on the papers?

    uv run python experiments/stage3/parser_vs_grobid.py <scratch dir>

Copies the database, the arXiv PDFs and the embedding cache into <scratch dir>, re-parses every
arXiv document there with the layout parser (rag/parse/pdf.py), and prints what changed per paper.
Then, by hand (each step is the normal command, pointed at the scratch copy):

    rag --data-dir <scratch> chunk
    rag --data-dir <scratch> index build
    rag --data-dir <scratch> eval check --set papers
    rag --data-dir <scratch> eval run --set papers --label layout
    rag eval run --set papers --label grobid            # the real data, unchanged
    rag eval compare <grobid run> <layout run>
"""

import json
import shutil
import sys
import time
from pathlib import Path

from rag.config import Settings
from rag.ingest import Ingestor
from rag.store import db


def main() -> int:
    scratch = Path(sys.argv[1]).resolve()
    real = Settings().data_dir
    if scratch.exists() and any(scratch.iterdir()):
        print(f"{scratch} is not empty; pass a new folder")
        return 1
    scratch.mkdir(parents=True, exist_ok=True)
    shutil.copy2(real / "rag.db", scratch / "rag.db")
    shutil.copytree(real / "raw" / "arxiv", scratch / "raw" / "arxiv")
    if (real / "embeddings").exists():
        shutil.copytree(real / "embeddings", scratch / "embeddings")

    settings = Settings(data_dir=scratch)
    conn = db.open_db(settings.db_path)
    before = {
        d.doc_id: (
            len(db.get_sections(conn, d.doc_id)),
            sum(len(s.text.split()) for s in db.get_sections(conn, d.doc_id)),
        )
        for d in db.list_documents(conn)
        if d.source == "arxiv"
    }
    documents = [db.get_document(conn, doc_id) for doc_id in before]
    conn.close()

    ingestor = Ingestor(settings, reparse=True, collection="papers", pdf_parser="pymupdf")
    rows = []
    start = time.perf_counter()
    for document in documents:
        pdf = scratch / "raw" / "arxiv" / f"{document.metadata['arxiv_id'].replace('/', '_')}.pdf"
        result = ingestor.parse(document, pdf)
        sections = db.get_sections(ingestor.conn, document.doc_id)
        rows.append(
            {
                "doc_id": document.doc_id,
                "status": result.status,
                "seconds": round(result.parse_seconds, 2),
                "grobid_sections": before[document.doc_id][0],
                "layout_sections": len(sections),
                "grobid_words": before[document.doc_id][1],
                "layout_words": sum(len(s.text.split()) for s in sections),
                "issues": result.issues,
            }
        )
    ingestor.close()
    out = Path(__file__).with_name("parser_vs_grobid.json")
    out.write_text(json.dumps({"parse_seconds_total": round(time.perf_counter() - start, 2), "papers": rows}, indent=1))
    for r in rows:
        print(
            f"{r['doc_id']:18} {r['status']:8} {r['seconds']:5.2f}s  sections {r['grobid_sections']:3} -> "
            f"{r['layout_sections']:3}  words {r['grobid_words']:6} -> {r['layout_words']:6}  {r['issues']}"
        )
    print(f"saved {out}; now chunk, index build and eval run with --data-dir {scratch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
