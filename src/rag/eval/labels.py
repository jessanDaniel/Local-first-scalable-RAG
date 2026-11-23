"""Manual labels: a Markdown sheet a person fills in, and the report that compares it with the
automatic checks.

The labeller never needs to know the papers. Each entry shows the question, the
expected answer, the REFERENCE PASSAGE (the gold quote that answers it, or "not in
the corpus"), the model's answer, and the text of every source it cited. Judge only
from what is shown:

    verdict: correct | partial | wrong | refused_ok | refused_wrong | unsure
    [n] supports: yes | partly | no        (one line per cited source)
    note: free text

Each entry starts with a hidden line <!-- entry: RUN_ID | QUESTION_ID | RUN --> so the
filled sheet can be matched back to the answer run.
"""

import re
from collections import Counter
from pathlib import Path

VERDICTS = ("correct", "partial", "wrong", "refused_ok", "refused_wrong", "unsure")
SUPPORTS = ("yes", "partly", "no")
ENTRY = re.compile(r"<!-- entry: (?P<run>\S+) \| (?P<qid>\S+) \| (?P<attempt>\d+) -->")
FIELD = re.compile(r"^(verdict|note):[ \t]*(.*)$", re.MULTILINE)
SUPPORT = re.compile(r"^\[(\d+)\] supports:[ \t]*(\S*)", re.MULTILINE)

GUIDE = """# Answer labels

For each entry, fill in the lines at the bottom. Judge **only from the text shown**: the
reference passage is what the paper says; the quoted sources are what the model was given.

| Field | Values | Meaning |
|---|---|---|
| `verdict` | `correct` | answers the question in agreement with the reference passage |
| | `partial` | right in part: misses something asked for, or adds a wrong/unsupported detail |
| | `wrong` | the main answer disagrees with the reference passage |
| | `refused_ok` | marked "not in the corpus", and the model said it could not answer |
| | `refused_wrong` | the reference passage answers it, but the model said the sources do not |
| | `unsure` | the passage shown is not enough to judge (say why in `note`) |
| `[n] supports` | `yes` / `partly` / `no` | does source [n]'s quoted text say what the sentences citing it claim? |
| `note` | free text | anything odd: often the most useful line |

An answer can be `correct` while a citation `supports: no` (right fact, wrong source). Mark both.

"""


def quote_block(text: str, limit: int | None = 1800) -> str:
    flat = " ".join(text.split())
    if limit is not None and len(flat) > limit:
        flat = flat[:limit] + " …"
    return "> " + flat


def render_entry(run_id: str, record: dict, attempt: dict, number: int) -> str:
    lines = [
        f"<!-- entry: {run_id} | {record['id']} | {attempt['run']} -->",
        f"## {number}. {record['id']} ({record['type']}, {record['collection']}), run {attempt['run']}",
        "",
        f"**Question:** {record['question']}",
        "",
    ]
    if record["answerable"]:
        lines += [f"**Expected answer:** {record['expected']}", "", "**Reference passage (what the paper says):**"]
        lines += [quote_block(q) for q in record["quotes"]]
    else:
        lines += [
            "**Expected answer:** *not in the corpus*: the right behaviour is to say the sources do not cover it."
        ]
    lines += ["", "**Model's answer:**", "", quote_block(attempt["answer"], limit=4000), ""]
    cited = [s for s in attempt["sources"] if s["cited"]]
    if cited:
        lines += ["**Sources the answer cites:**", ""]
        for source in cited:
            # Sources are shown in full: a cut-off quote can hide the very sentence that supports a
            # claim (found on the first sheet), and the labeller may only judge from what is shown.
            lines += [f"**{source['label']}**", quote_block(source["text"], limit=None), ""]
    else:
        lines += ["*(the answer cites no source)*", ""]
    lines += ["verdict: "] + [f"[{s['n']}] supports: " for s in cited] + ["note: ", "", "---", ""]
    return "\n".join(lines)


def pick_entries(runs: list[dict], per_run: dict[str, int]) -> list[tuple[dict, dict, dict]]:
    """A varied sample: per run, round-robin over the automatic outcomes (so every kind appears),
    question types spread out, plus any question whose repeat runs disagreed. Deterministic."""
    picked = []
    for run in runs:
        want = per_run.get(run["set"], 0)
        records = run["questions"]
        unstable = [r for r in records if len(r["attempts"]) > 1 and not r["consistency"]["same_outcome"]]
        chosen: list[tuple[dict, dict]] = [(r, r["attempts"][-1]) for r in unstable[:1]]  # the differing repeat
        by_outcome: dict[str, list[dict]] = {}
        for record in sorted(records, key=lambda r: (r["type"], r["id"])):
            by_outcome.setdefault(record["attempts"][0]["outcome"], []).append(record)
        queues = [list(v) for _, v in sorted(by_outcome.items())]
        types_seen: Counter = Counter()
        while len(chosen) < want and any(queues):
            for queue in queues:
                if not queue or len(chosen) >= want:
                    continue
                queue.sort(key=lambda r: (types_seen[r["type"]], r["type"], r["id"]))
                record = queue.pop(0)
                if any(record is c for c, _ in chosen):
                    continue
                chosen.append((record, record["attempts"][0]))
                types_seen[record["type"]] += 1
        picked += [(run, record, attempt) for record, attempt in chosen]
    return picked


def write_sheet(runs: list[dict], per_run: dict[str, int], path: Path) -> int:
    entries = pick_entries(runs, per_run)
    body = [render_entry(run["run_id"], record, attempt, i) for i, (run, record, attempt) in enumerate(entries, 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GUIDE + "\n".join(body))
    return len(entries)


def parse_sheet(text: str) -> tuple[list[dict], list[str]]:
    """Filled sheet -> labels, plus problems (missing or invalid values)."""
    labels, problems = [], []
    starts = list(ENTRY.finditer(text))
    for i, match in enumerate(starts):
        chunk = text[match.end() : starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        fields = {name: value.strip() for name, value in FIELD.findall(chunk)}
        supports = {int(n): value.strip().lower() for n, value in SUPPORT.findall(chunk)}
        qid = match["qid"]
        verdict = fields.get("verdict", "").lower()
        if verdict not in VERDICTS:
            problems.append(f"{qid}: verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
        for n, value in supports.items():
            if value not in SUPPORTS:
                problems.append(f"{qid}: [{n}] supports {value!r} is not one of {', '.join(SUPPORTS)}")
        labels.append(
            {
                "run_id": match["run"],
                "id": qid,
                "run": int(match["attempt"]),
                "verdict": verdict,
                "supports": supports,
                "note": fields.get("note", ""),
            }
        )
    return labels, problems


AGREES = {  # automatic outcome -> manual verdicts that count as agreement
    "correct?": {"correct"},
    "wrong?": {"wrong", "partial"},
    "refused_wrong": {"refused_wrong"},
    "refused_ok": {"refused_ok"},
    "answered?": {"correct", "partial", "wrong"},  # answered an unanswerable question
}


def agreement_report(labels: list[dict], runs: dict[str, dict]) -> list[str]:
    """How often each automatic check agrees with the person's labels."""
    rows = []
    for label in labels:
        if label["verdict"] not in VERDICTS or label["verdict"] == "unsure":
            continue
        record = next(r for r in runs[label["run_id"]]["questions"] if r["id"] == label["id"])
        attempt = next(a for a in record["attempts"] if a["run"] == label["run"])
        rows.append((label, record, attempt))
    lines = [f"labelled answers: {len(labels)} ({sum(lb['verdict'] == 'unsure' for lb in labels)} unsure, excluded)"]
    if not rows:
        return lines
    verdicts = Counter(label["verdict"] for label, _, _ in rows)
    lines.append("verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))

    outcome_ok = sum(label["verdict"] in AGREES[a["outcome"]] for label, _, a in rows)
    lines.append(f"automatic outcome agrees with the label: {outcome_ok}/{len(rows)}")
    keyed = [
        (label, a)
        for label, r, a in rows
        if r["answerable"] and a["key_match"] is not None and label["verdict"] in ("correct", "partial", "wrong")
    ]
    if keyed:
        hit = sum(a["key_match"] == (label["verdict"] == "correct") for label, a in keyed)
        lines.append(f"key_match agrees (match <-> labelled correct): {hit}/{len(keyed)}")
    refusal_ok = sum(a["refused"] == label["verdict"].startswith("refused") for label, _, a in rows)
    lines.append(f"refusal detector agrees: {refusal_ok}/{len(rows)}")

    supports = Counter(value for label, _, _ in rows for value in label["supports"].values())
    total = sum(supports.values())
    if total:
        lines.append(
            f"cited sources checked: {total}; supports yes {supports['yes']}, partly {supports['partly']}, "
            f"no {supports['no']}"
        )
        unsupported_but_valid = sum(
            1 for label, _, a in rows for n, value in label["supports"].items() if value == "no" and n in a["cited"]
        )
        lines.append(
            f"citations that pass the automatic check (number exists) but do not support the claim: "
            f"{unsupported_but_valid}"
        )
    disagreements = [(label, a) for label, _, a in rows if label["verdict"] not in AGREES[a["outcome"]]]
    if disagreements:
        lines.append("disagreements (automatic -> label):")
        lines += [
            f"  {label['id']} run {label['run']}: {a['outcome']} -> {label['verdict']}"
            + (f"  ({label['note']})" if label["note"] else "")
            for label, a in disagreements
        ]
    return lines
