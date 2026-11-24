"""Step 21 follow-up: replay every stored answer through the guardrails before and after the
refusal-sentence fix and list every answer whose shown text changed.

    uv run python experiments/stage3/refusal_text_check.py <old guardrails.py>
"""

import importlib.util
import json
import sys
from pathlib import Path

from rag.answer import guardrails as new
from rag.config import get_settings


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("guardrails_old", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    old = load_old(sys.argv[1])
    answers_dir = get_settings().data_dir / "eval" / "answers"
    total = changed = refusals = 0
    rows = []
    for path in sorted(answers_dir.glob("*.json")):
        run = json.loads(path.read_text())
        for record in run.get("questions", []):
            for attempt in record.get("attempts", []):
                raw = attempt.get("raw_answer") or attempt.get("answer") or ""
                total += 1
                before, n_before = old.strip_refusal_citations(raw)
                after, n_after = new.strip_refusal_citations(raw)
                refusals += new.is_refusal(raw)
                if before != after:
                    changed += 1
                    rows.append(
                        {
                            "run": run.get("run_id"),
                            "id": record.get("id"),
                            "before": before,
                            "after": after,
                            "removed_before": n_before,
                            "removed_after": n_after,
                        }
                    )
    out = {"answers": total, "refusals": refusals, "text_changed": changed, "changes": rows}
    Path(__file__).with_name("refusal_text_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{total} stored answers, {refusals} refusals, shown text changed on {changed}")
    for r in rows:
        print(f"\n--- {r['run']} {r['id']} (markers removed {r['removed_before']} -> {r['removed_after']})")
        print("BEFORE:", r["before"][:600])
        print("AFTER: ", r["after"][:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
