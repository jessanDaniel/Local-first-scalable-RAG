"""Phase 15 — check that every eval quote actually occurs in its document.

Whitespace is collapsed before matching, because several facts wrap across lines
in the source files. A quote must occur exactly once. Missing-information rows
must have a null answer and no quotes.
"""

import json
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

TYPES = {"direct", "multi_step", "missing_information", "distractor"}


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    path = project_root / "evaluations" / "eval_questions.jsonl"
    docs: dict[str, str] = {}
    rows = []
    errors = []

    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        if row["type"] not in TYPES:
            errors.append(f"line {line_no}: unknown type {row['type']}")
        if row["doc"] not in docs:
            docs[row["doc"]] = normalize((project_root / "documents" / row["doc"]).read_text())
        doc = docs[row["doc"]]

        if row["type"] == "missing_information":
            if row["expected_answer"] is not None or row["source_quotes"] or row["distractor_quotes"]:
                errors.append(f"{row['id']}: missing-information row must have a null answer and no quotes")
            continue

        if not row["expected_answer"] or not row["source_quotes"]:
            errors.append(f"{row['id']}: answerable row needs expected_answer and source_quotes")
        if row["type"] == "distractor" and not row["distractor_quotes"]:
            errors.append(f"{row['id']}: distractor row needs distractor_quotes")

        for field in ("source_quotes", "distractor_quotes"):
            for quote in row[field]:
                count = doc.count(normalize(quote))
                if count != 1:
                    errors.append(f"{row['id']}: {field} occurs {count} times: {quote}")

    counts = Counter(row["type"] for row in rows)
    print(f"{len(rows)} questions")
    for name in ("direct", "multi_step", "missing_information", "distractor"):
        print(f"  {name}: {counts[name]}")
    if errors:
        print(f"\n{len(errors)} problems")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    print("every quote occurs once in its document")


if __name__ == "__main__":
    main()
