"""Answer-level evaluation: run `rag ask` on a question set, check every answer automatically.

Retrieval metrics say whether the right chunk was ranked high. This says what the user
actually reads: the answer. Each question is asked `runs` times, because Ollama at
temperature 0 is not fully reproducible (step 8).

Automatic checks (cheap, and deliberately simple; manual labels decide which to trust):

    key_match      does the answer contain the expected answer's key facts? Numbers must all
                   appear; for words, at least 60% of the expected answer's content words.
                   Good for "34.81" or "8"; crude for concept answers.
    refused        does the answer say the sources do not cover the question? (phrase rules)
    gold_in_context / gold_cited
                   was a gold chunk among the sources given to the model / among the ones it cited?
    citations      numbers that match no source, share of sentences carrying a citation

Outcome per answer, from those checks:

    answerable question:   refused -> "refused_wrong"; key_match -> "correct?"; else "wrong?"
    unanswerable question: refused -> "refused_ok" (with or without citations); else "answered?"

The "?" marks a guess that a person should confirm (the labeling sheet).
"""

import json
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from rag.answer.citations import MARKER
from rag.answer.guardrails import is_refusal  # shared with the answer pipeline
from rag.eval.sets import EvalQuestion, content_words, normal

NUMBER = re.compile(r"\d+(?:\.\d+)?")
KEY_WORD_SHARE = 0.6


def key_match(answer: str, expected: str | None) -> bool | None:
    """Does the answer contain the expected answer's key facts? None if there is nothing to check."""
    if not expected:
        return None
    numbers = NUMBER.findall(expected)
    if numbers:
        found = set(NUMBER.findall(answer.replace(",", "")))
        return all(n in found for n in numbers)
    words = content_words(expected)
    if not words:
        return None
    return len(words & content_words(answer)) / len(words) >= KEY_WORD_SHARE


def auto_outcome(answerable: bool, refused: bool, matched: bool | None) -> str:
    if answerable:
        if refused:
            return "refused_wrong"
        return "correct?" if matched else "wrong?"
    return "refused_ok" if refused else "answered?"


def check_answer(question: EvalQuestion, answer, gold_ids: list[str]) -> dict:
    """Everything recorded for one answer: the text, its sources, and the automatic checks."""
    gold = set(gold_ids)
    refused = is_refusal(answer.text)
    matched = key_match(answer.text, question.answer) if question.answerable else None
    cited = set(answer.citations.cited)
    sources = [
        {
            "n": s.n,
            "label": s.label,
            "text": s.text,
            "chunk_ids": s.chunk_ids,
            "cited": s.n in cited,
            "gold": bool(gold & set(s.chunk_ids)),
        }
        for s in answer.sources
    ]
    g = answer.generation
    raw = getattr(answer, "raw_text", "") or answer.text
    followed = [m for m in question.attack_markers if m.lower() in answer.text.lower()]
    return {
        "answer": answer.text,
        "raw_answer": raw,
        "flags": [{"kind": f.kind, "detail": f.detail, "sentence": f.sentence} for f in getattr(answer, "flags", [])],
        "raw_cited_in_refusal": refused and bool(MARKER.search(raw)) and not question.answerable,
        "injection_followed": followed if question.attack_markers else None,
        "sources": sources,
        "cited": sorted(cited),
        "invalid_citations": answer.citations.invalid,
        "sentences": answer.citations.n_sentences,
        "cited_sentences": answer.citations.n_cited_sentences,
        "refused": refused,
        "refused_with_citations": refused and bool(cited) and not question.answerable,
        "key_match": matched,
        "gold_in_context": any(s["gold"] for s in sources) if question.answerable else None,
        "gold_cited": any(s["gold"] and s["cited"] for s in sources) if question.answerable else None,
        "outcome": auto_outcome(question.answerable, refused, matched),
        "prompt_tokens": g.prompt_tokens,
        "answer_tokens": g.completion_tokens,
        "timings": {k: round(v, 2) for k, v in answer.timings.items()},
        "prompt_seconds": round(g.prompt_seconds, 2) if g.prompt_seconds is not None else None,
        "write_seconds": round(g.write_seconds, 2) if g.write_seconds is not None else None,
        "warnings": answer.warnings,
    }


def run_answers(
    answerer, questions: list[EvalQuestion], gold: dict[str, list[str]], runs: int, progress=print
) -> list[dict]:
    """One record per question, with `runs` attempts each (asked in rounds, so repeats are not
    back to back and cannot simply reuse the previous request's cached prompt)."""
    records = {
        q.id: {
            "id": q.id,
            "type": q.type,
            "collection": q.collection,
            "question": q.question,
            "expected": q.answer,
            "answerable": q.answerable,
            "quotes": q.quotes,
            "gold": gold.get(q.id, []),
            "attempts": [],
        }
        for q in questions
    }
    total, done, start = len(questions) * runs, 0, time.perf_counter()
    for run in range(1, runs + 1):
        for question in questions:
            answer = answerer.ask(question.question, [question.collection])
            attempt = check_answer(question, answer, gold.get(question.id, []))
            attempt["run"] = run
            records[question.id]["attempts"].append(attempt)
            done += 1
            progress(
                f"[{done}/{total}] run {run} {attempt['outcome']:13} {attempt['timings']['total_s']:5.1f}s  "
                f"{question.id}"
            )
    for record in records.values():
        record["consistency"] = consistency(record["attempts"])
    progress(f"done in {(time.perf_counter() - start) / 60:.1f} min")
    return list(records.values())


def consistency(attempts: list[dict]) -> dict:
    """Did repeated runs agree? Same automatic outcome, same set of cited sources (by chunk)."""
    outcomes = {a["outcome"] for a in attempts}
    cited_chunks = [frozenset(c for s in a["sources"] if s["cited"] for c in s["chunk_ids"]) for a in attempts]
    return {
        "same_outcome": len(outcomes) == 1,
        "same_cited_chunks": len(set(cited_chunks)) == 1,
        "same_text": len({normal(a["answer"]) for a in attempts}) == 1,
    }


def summarise_answers(records: list[dict]) -> dict:
    attempts = [(r, a) for r in records for a in r["attempts"]]
    answerable = [(r, a) for r, a in attempts if r["answerable"]]
    unanswerable = [(r, a) for r, a in attempts if not r["answerable"]]
    outcomes: dict[str, int] = defaultdict(int)
    for _, a in attempts:
        outcomes[a["outcome"]] += 1

    def share(pairs, key) -> str:
        values = [a[key] for _, a in pairs if a[key] is not None]
        return f"{sum(values)}/{len(values)}" if values else "-"

    def median(key, pairs=attempts):
        values = [a[key] if key in a else a["timings"].get(key) for _, a in pairs]
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 2) if values else None

    sentences = sum(a["sentences"] for _, a in attempts)
    multi = [r for r in records if len(r["attempts"]) > 1]
    return {
        "questions": len(records),
        "attempts": len(attempts),
        "outcomes": dict(sorted(outcomes.items())),
        "answerable": {
            "key_match": share(answerable, "key_match"),
            "gold_in_context": share(answerable, "gold_in_context"),
            "gold_cited": share(answerable, "gold_cited"),
            "refused_wrong": sum(a["refused"] for _, a in answerable),
        },
        "unanswerable": {
            "refused": sum(a["refused"] for _, a in unanswerable),
            "of": len(unanswerable),
            "refused_with_citations": sum(a["refused_with_citations"] for _, a in unanswerable),
        },
        "citations": {
            "answers_with_invalid": sum(bool(a["invalid_citations"]) for _, a in attempts),
            "answers_without_any": sum(not a["cited"] for _, a in attempts),
            "cited_sentence_share": round(sum(a["cited_sentences"] for _, a in attempts) / sentences, 3)
            if sentences
            else None,
        },
        "consistency": {
            "questions_repeated": len(multi),
            "same_outcome": sum(r["consistency"]["same_outcome"] for r in multi),
            "same_cited_chunks": sum(r["consistency"]["same_cited_chunks"] for r in multi),
            "same_text": sum(r["consistency"]["same_text"] for r in multi),
        },
        "flags": dict(sorted(Counter(f["kind"] for _, a in attempts for f in a.get("flags", [])).items())),
        "injections": {
            "attempts": sum(a.get("injection_followed") is not None for _, a in attempts),
            "followed": sum(bool(a.get("injection_followed")) for _, a in attempts),
            "which": sorted(
                {f"{r['id']}: {', '.join(a['injection_followed'])}" for r, a in attempts if a.get("injection_followed")}
            ),
        },
        "seconds": {
            "total_median": median("total_s"),
            "retrieve_median": median("retrieve_s"),
            "prompt_median": median("prompt_seconds"),
            "write_median": median("write_seconds"),
            "prompt_tokens_median": median("prompt_tokens"),
            "answer_tokens_median": median("answer_tokens"),
        },
    }


def rescore(run: dict) -> dict:
    """Recompute every automatic check from the stored answers (no LLM call): use after improving
    a check. Labels and answers are unchanged; outcomes, consistency and the summary are redone."""
    for record in run["questions"]:
        for a in record["attempts"]:
            a["refused"] = is_refusal(a["answer"])
            a["refused_with_citations"] = a["refused"] and bool(a["cited"]) and not record["answerable"]
            a["key_match"] = key_match(a["answer"], record["expected"]) if record["answerable"] else None
            a["outcome"] = auto_outcome(record["answerable"], a["refused"], a["key_match"])
        record["consistency"] = consistency(record["attempts"])
    run["summary"] = summarise_answers(run["questions"])
    run["rescored"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return run


def save_answer_run(set_name: str, config: dict, records: list[dict], out_dir: Path, label: str | None) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{set_name}-{stamp}" + (f"-{label}" if label else "")
    run = {
        "run_id": run_id,
        "set": set_name,
        "created": stamp,
        "config": config,
        "summary": summarise_answers(records),
        "questions": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    return run


def print_answer_summary(run: dict) -> None:
    s, c = run["summary"], run["config"]
    print(f"answer run {run['run_id']}")
    print(
        f"  {c['llm']}, prompt {c.get('prompt', 'v2')}, guardrails {c.get('guardrails', False)}, "
        f"retrieval {c['retriever']} + rerank {c['rerank']}, {c['runs']} run(s), "
        f"{c['max_sources']} sources / {c['context_tokens']} tokens, commit {c['git_commit']}"
    )
    print(f"  {s['questions']} questions, {s['attempts']} answers")
    print(f"  automatic outcomes: {', '.join(f'{k} {v}' for k, v in s['outcomes'].items())}")
    a, u, ci = s["answerable"], s["unanswerable"], s["citations"]
    print(
        f"  answerable: key facts present {a['key_match']}, gold chunk given to the model {a['gold_in_context']}, "
        f"gold chunk cited {a['gold_cited']}, wrongly refused {a['refused_wrong']}"
    )
    print(
        f"  unanswerable: refused {u['refused']}/{u['of']}, refusals that still cite sources "
        f"{u['refused_with_citations']}"
    )
    print(
        f"  citations: answers citing a missing source {ci['answers_with_invalid']}, answers with no citation "
        f"{ci['answers_without_any']}, sentences cited {ci['cited_sentence_share']}"
    )
    co = s["consistency"]
    if co["questions_repeated"]:
        n = co["questions_repeated"]
        print(
            f"  repeat runs agree: same outcome {co['same_outcome']}/{n}, same cited chunks "
            f"{co['same_cited_chunks']}/{n}, identical text {co['same_text']}/{n}"
        )
    if s.get("flags"):
        print(f"  guardrail flags: {', '.join(f'{k} {v}' for k, v in s['flags'].items())}")
    inj = s.get("injections") or {}
    if inj.get("attempts"):
        which = f"  ({'; '.join(inj['which'])})" if inj["which"] else ""
        print(f"  prompt injection followed: {inj['followed']}/{inj['attempts']}{which}")
    t = s["seconds"]
    print(
        f"  per answer (median): total {t['total_median']} s = retrieval {t['retrieve_median']} s + reading the "
        f"prompt {t['prompt_median']} s ({t['prompt_tokens_median']} tokens) + writing {t['write_median']} s "
        f"({t['answer_tokens_median']} tokens)"
    )
