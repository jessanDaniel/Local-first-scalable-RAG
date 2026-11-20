"""Stage 1 step 7 smoke check: does dense search find the right chunk, and do the
query instruction and the title/section header earn their place?

Not the evaluation (still deferred): 10 paraphrased questions, one per paper, each with
the chunk(s) that answer it. Reports the rank of the best correct chunk out of all 400 for:

    A  full design: query instruction + "title | section path" header on passages
    B  no query instruction
    C  no header (passages embedded as bare chunk text)

Also checks that FAISS returns exactly what a NumPy dot product over the same vectors gives.

    uv run python experiments/stage1/dense_smoke.py
"""

import csv
import time
from pathlib import Path

import numpy as np

from rag.chunk.tokens import HFTokenCounter
from rag.config import get_settings
from rag.embed.embedder import Embedder, passage_text
from rag.index.dense import load_current
from rag.store import db

OUT = Path(__file__).with_name("dense_smoke.csv")

# (question in our own words, doc_id, accepted section-path prefixes, accepted text phrases)
QUESTIONS = [
    (
        "why divide attention scores by the square root of the dimension",
        "arxiv:1706.03762",
        [],
        ["grow large in magnitude"],
    ),
    (
        "correcting moving averages of the gradient that start at zero",
        "arxiv:1412.6980",
        ["3 INITIALIZATION BIAS CORRECTION"],
        ["biased towards zero"],
    ),
    (
        "train a small network on the softened outputs of a big model",
        "arxiv:1503.02531",
        ["2 Distillation"],
        ["soft targets"],
    ),
    (
        "how to backpropagate through random sampling of a latent variable",
        "arxiv:1312.6114",
        ["2 Method > 2.4"],
        ["reparameterization trick"],
    ),
    (
        "why very deep networks get worse and how learning a residual fixes it",
        "arxiv:1512.03385",
        ["3 Deep Residual Learning > 3.1"],
        ["residual mapping", "degradation problem"],
    ),
    (
        "generate images by learning to undo noise that was added step by step",
        "arxiv:2006.11239",
        [],
        ["reverse process"],
    ),
    (
        "reversing the order of the words in the input sentence helps the LSTM",
        "arxiv:1409.3215",
        ["3 Experiments > 3.3"],
        [],
    ),
    (
        "two networks trained against each other, one making fakes and one spotting them",
        "arxiv:1406.2661",
        ["3 Adversarial nets"],
        ["minimax"],
    ),
    (
        "layer inputs keep changing distribution during training, which slows learning",
        "arxiv:1502.03167",
        [],
        ["internal covariate shift"],
    ),
    ("predict the surrounding words given the current word", "arxiv:1301.3781", ["3 New Log-linear Models > 3.2"], []),
]


def main() -> None:
    settings = get_settings()
    conn = db.open_db(settings.db_path)
    counter = HFTokenCounter(settings.embedding_model)
    embedder = Embedder(settings.embedding_model, settings.embedding_query_instruction, settings.embedding_batch_size)

    rows = []  # (chunk, doc_title, section_path)
    for document in db.list_documents(conn):
        paths = {s.ord: s.path for s in db.get_sections(conn, document.doc_id)}
        rows.extend((c, document.title, paths[c.section_ord]) for c in db.get_chunks(conn, document.doc_id))
    chunk_ids = [c.chunk_id for c, _, _ in rows]

    start = time.perf_counter()
    with_header = embedder.embed_passages([passage_text(t, p, c.text, counter) for c, t, p in rows])
    without_header = embedder.embed_passages([c.text for c, _, _ in rows])
    print(f"embedded {len(rows)} chunks twice in {time.perf_counter() - start:.1f}s on {embedder.device}")

    # 1. FAISS vs NumPy on the stored index.
    index = load_current(settings.data_dir / "indexes" / "dense", settings.embedding_model, db.all_chunk_ids(conn))
    stored = np.stack([with_header[chunk_ids.index(cid)] for cid in index.chunk_ids])
    worst = 0.0
    for question, *_ in QUESTIONS:
        query = embedder.embed_query(question)
        faiss_hits = index.search(query, 10)
        numpy_scores = stored @ query
        numpy_top = np.argsort(-numpy_scores)[:10]
        assert [cid for cid, _ in faiss_hits] == [index.chunk_ids[i] for i in numpy_top]
        worst = max(worst, float(np.max(np.abs(np.array([s for _, s in faiss_hits]) - numpy_scores[numpy_top]))))
    print(f"FAISS top-10 == NumPy top-10 for all {len(QUESTIONS)} questions; max score difference {worst:.2e}")

    # 2. Rank of the best correct chunk under each variant.
    def correct(i: int, doc_id: str, paths: list[str], phrases: list[str]) -> bool:
        chunk, _, path = rows[i]
        text = " ".join(chunk.text.split()).lower()
        return chunk.doc_id == doc_id and (
            any(path.startswith(p) for p in paths) or any(p.lower() in text for p in phrases)
        )

    variants = {
        "A_full": (with_header, True),
        "B_no_instruction": (with_header, False),
        "C_no_header": (without_header, True),
    }
    results = []
    for question, doc_id, paths, phrases in QUESTIONS:
        accepted = [i for i in range(len(rows)) if correct(i, doc_id, paths, phrases)]
        assert accepted, f"no chunk matches the answer spec for {question!r}"
        row = {"question": question, "doc_id": doc_id, "n_accepted": len(accepted)}
        for name, (vectors, instruction) in variants.items():
            query = embedder.embed_query(question) if instruction else embedder.embed_passages([question])[0]
            order = np.argsort(-(vectors @ query))
            rank = 1 + min(int(np.where(order == i)[0][0]) for i in accepted)
            row[f"{name}_rank"] = rank
            row[f"{name}_top1_doc_ok"] = rows[order[0]][0].doc_id == doc_id
        results.append(row)

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'question':72} {'A':>4} {'B':>4} {'C':>4}")
    for r in results:
        print(
            f"{r['question'][:72]:72} {r['A_full_rank']:>4} {r['B_no_instruction_rank']:>4} {r['C_no_header_rank']:>4}"
        )
    for name in variants:
        ranks = [r[f"{name}_rank"] for r in results]
        print(
            f"{name:18} rank1 {sum(r == 1 for r in ranks)}/10  top3 {sum(r <= 3 for r in ranks)}/10  "
            f"top5 {sum(r <= 5 for r in ranks)}/10  mean rank {np.mean(ranks):.1f}  "
            f"top-1 from right paper {sum(r[f'{name}_top1_doc_ok'] for r in results)}/10"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
