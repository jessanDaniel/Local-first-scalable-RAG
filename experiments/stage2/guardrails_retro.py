"""Step 17 layer A on the step-16 answers, without calling the model.

Applies rag/answer/guardrails.py to every stored answer (papers 2 runs + hr-demo 1 run, prompt v2,
no guardrails at the time) and reports:
- how many refusal citations would have been stripped,
- which flags fire, per kind,
- for the 15 answers labelled in step 16: does each failure get flagged, and does any answer
  labelled correct get a flag (a false flag)?

    uv run python experiments/stage2/guardrails_retro.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from rag.answer.guardrails import check_answer_text
from rag.config import get_settings
from rag.eval.labels import parse_sheet
from rag.store import db

RUNS = ["papers-20261123T062859Z-fused", "hr-demo-20261123T063124Z-fused"]
SHEET = Path("eval/labels/answers-fused-2026-11-23.md")


def main() -> None:
    settings = get_settings()
    conn = db.open_db(settings.db_path)
    doc_cache: dict[str, str] = {}

    def doc_text(doc_id: str) -> str:
        if doc_id not in doc_cache:
            document = db.get_document(conn, doc_id)
            doc_cache[doc_id] = f"{document.title} " + " ".join(s.text for s in db.get_sections(conn, doc_id))
        return doc_cache[doc_id]

    labels = {(lb["run_id"], lb["id"], lb["run"]): lb for lb in parse_sheet(SHEET.read_text())[0]}
    kinds, stripped, refusals, answers = Counter(), 0, 0, 0
    labelled_rows = []
    for run_id in RUNS:
        run = json.loads((settings.data_dir / "eval" / "answers" / f"{run_id}.json").read_text())
        for record in run["questions"]:
            for attempt in record["attempts"]:
                sources = attempt["sources"]
                first_chunks = [s["chunk_ids"][0] for s in sources]
                chunk_doc = {c.chunk_id: c.doc_id for c in db.get_chunks_by_ids(conn, first_chunks)}
                source_docs = {s["n"]: chunk_doc[s["chunk_ids"][0]] for s in sources}
                result = check_answer_text(
                    record["question"],
                    attempt["answer"],
                    {s["n"]: s["text"] for s in sources},
                    source_docs,
                    {d: doc_text(d) for d in set(source_docs.values())},
                )
                answers += 1
                refusals += result.refusal
                stripped += result.removed_citations
                kinds.update(f.kind for f in result.flags)
                key = (run_id, record["id"], attempt["run"])
                if key in labels:
                    labelled_rows.append((record["id"], labels[key]["verdict"], result))

    print(f"answers: {answers}, refusals: {refusals}, refusal citations that would be stripped: {stripped}")
    print("flags: " + (", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none"))
    print("\nthe 15 labelled answers:")
    for qid, verdict, result in labelled_rows:
        flags = "; ".join(f"{f.kind}: {f.detail}" for f in result.flags) or "-"
        print(f"  {verdict:11} {qid:28} {flags[:150]}")
    false_flags = [(q, r) for q, v, r in labelled_rows if v in ("correct", "refused_ok") and r.flags]
    print(f"\nflags on answers labelled correct or refused_ok (false flags): {len(false_flags)}")
    for qid, result in false_flags:
        for flag in result.flags:
            print(f"  {qid}: {flag.kind}: {flag.detail}\n      {re.sub(r'\\s+', ' ', flag.sentence)[:200]}")


if __name__ == "__main__":
    main()
