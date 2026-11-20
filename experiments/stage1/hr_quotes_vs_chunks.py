"""Step 11b check: do the Phase 15 eval quotes survive the production chunker?

Phase 15 wrote 30 questions about the two HR documents, each answerable one with the
exact `source_quotes` that answer it. Now those documents are in the `hr-demo`
collection, chunked by chunker v2. For every quote, find the chunks of its document
that contain it (whitespace-normalised):

    in one chunk     -> that chunk is the gold answer for eval v0
    split            -> the quote crosses a chunk boundary (the Phase 14 failure)
    missing          -> the text changed (parse bug)

Also, per answerable question: is a gold chunk ranked 1st by dense search with the
hr-demo filter, and without it (all 429 chunks, papers included)? Not the eval: no
answers are generated and only top-1 is reported.

    uv run python experiments/stage1/hr_quotes_vs_chunks.py   # -> hr_quotes_vs_chunks.csv + _gold.jsonl
"""

import csv
import json
from pathlib import Path

from rag.config import get_settings
from rag.embed.embedder import Embedder
from rag.index.dense import load_current
from rag.search import DenseSearcher
from rag.store import db

HERE = Path(__file__).parent
QUESTIONS = HERE.parents[1] / "experiments" / "learning" / "eval_questions.jsonl"


def normal(text: str) -> str:
    return " ".join(text.split()).lower()


def main() -> None:
    settings = get_settings()
    conn = db.open_db(settings.db_path)
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]

    quote_rows, gold = [], []
    for q in questions:
        doc_id = "file:" + Path(q["doc"]).stem
        chunks = db.get_chunks(conn, doc_id)
        assert chunks, f"{doc_id} not ingested: rag ingest files documents/ --collection hr-demo"
        gold_ids: set[str] = set()
        for kind in ("source_quotes", "distractor_quotes"):
            for quote in q.get(kind) or []:
                holders = [c.chunk_id for c in chunks if normal(quote) in normal(c.text)]
                status = "one chunk" if len(holders) == 1 else ("missing" if not holders else "several chunks")
                if not holders:
                    # Split across a boundary? The quote's first and last thirds sit in different chunks.
                    words = quote.split()
                    head, tail = normal(" ".join(words[:3])), normal(" ".join(words[-3:]))
                    if any(head in normal(c.text) for c in chunks) and any(tail in normal(c.text) for c in chunks):
                        status = "split"
                quote_rows.append(
                    {"id": q["id"], "kind": kind, "quote": quote, "status": status, "chunk_ids": " ".join(holders)}
                )
                if kind == "source_quotes":
                    gold_ids.update(holders)
        gold.append(
            {
                "id": q["id"],
                "type": q["type"],
                "doc_id": doc_id,
                "question": q["question"],
                "gold_chunk_ids": sorted(gold_ids),
            }
        )

    index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model, conn)
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)
    searcher = DenseSearcher(conn, index, embedder)
    for g in gold:
        if not g["gold_chunk_ids"]:
            continue
        for label, collections in (("filtered", ["hr-demo"]), ("unfiltered", None)):
            top = searcher.search(g["question"], 1, collections)[0]
            g[f"top1_{label}"] = top.chunk.chunk_id in g["gold_chunk_ids"]
            g[f"top1_{label}_doc"] = top.chunk.doc_id

    with (HERE / "hr_quotes_vs_chunks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quote_rows[0]))
        writer.writeheader()
        writer.writerows(quote_rows)
    with (HERE / "hr_quotes_vs_chunks_gold.jsonl").open("w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")

    for kind in ("source_quotes", "distractor_quotes"):
        rows = [r for r in quote_rows if r["kind"] == kind]
        counts = {s: sum(r["status"] == s for r in rows) for s in ("one chunk", "several chunks", "split", "missing")}
        print(f"{kind:18} {len(rows):3} quotes: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    answerable = [g for g in gold if g["gold_chunk_ids"]]
    print(f"questions: {len(gold)} ({len(answerable)} answerable, {len(gold) - len(answerable)} with no answer)")
    for label in ("filtered", "unfiltered"):
        hits = sum(g[f"top1_{label}"] for g in answerable)
        off = sum(not g[f"top1_{label}_doc"].startswith("file:") for g in answerable)
        print(f"top-1 is a gold chunk, {label:10}: {hits}/{len(answerable)}  (top-1 from a paper: {off})")
    for g in answerable:
        if g["top1_filtered"] != g["top1_unfiltered"]:
            print(
                f"  differs: {g['id']}: filtered {g['top1_filtered']}, unfiltered {g['top1_unfiltered']} "
                f"({g['top1_unfiltered_doc']})"
            )


if __name__ == "__main__":
    main()
