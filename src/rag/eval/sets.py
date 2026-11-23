"""Evaluation question sets: JSONL files in eval/sets/, gold answers stored as quotes.

One question per line:

    {"id": "attn_heads", "collection": "papers", "doc_id": "arxiv:1706.03762",
     "type": "identifier", "question": "How many attention heads ...?",
     "quotes": ["In this work we employ h = 8 parallel attention layers, or heads."],
     "answer": "8", "tags": []}

Gold is the exact text that answers the question, not a chunk id. The chunk(s)
holding each quote are found when the set is scored, so re-chunking never
invalidates a set. A question with no quotes (and no doc_id) is unanswerable from
the corpus: it is kept for answer evaluation (should the system refuse?) but has
no retrieval score.
"""

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import PROJECT_ROOT
from rag.store import db

SETS_DIR = PROJECT_ROOT / "eval" / "sets"
STOPWORDS = set(
    "a an and are as at be by can do does for from has have how in into is it its of on or that the their this "
    "to was were what when which who why with".split()
)


@dataclass
class EvalQuestion:
    id: str
    collection: str
    question: str
    type: str
    doc_id: str | None = None
    quotes: list[str] = field(default_factory=list)
    answer: str | None = None
    tags: list[str] = field(default_factory=list)
    distractor_quotes: list[str] = field(default_factory=list)
    # Prompt-injection questions: strings whose presence in the answer means a planted instruction
    # was followed (e.g. "HACKED", "evil.example").
    attack_markers: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.quotes)


@dataclass
class QuoteCheck:
    question_id: str
    quote: str
    status: str  # one chunk | several chunks | split | missing
    chunk_ids: list[str]


def set_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".jsonl":
        return path
    return SETS_DIR / f"{name_or_path}.jsonl"


def load_set(path: Path) -> list[EvalQuestion]:
    questions, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion(**json.loads(line))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        if question.id in seen:
            raise ValueError(f"{path.name} line {number}: duplicate id {question.id!r}")
        if bool(question.quotes) != bool(question.doc_id):
            raise ValueError(f"{path.name} line {number}: {question.id} needs both doc_id and quotes, or neither")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise ValueError(f"{path.name} has no questions")
    return questions


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def normal(text: str) -> str:
    """Whitespace-insensitive form used to match quotes against chunk text."""
    return " ".join(text.split())


def find_quote(quote: str, chunks: list[tuple[str, str]]) -> QuoteCheck:
    """chunks: (chunk_id, text) of the quote's document."""
    target = normal(quote)
    holders = [chunk_id for chunk_id, text in chunks if target in normal(text)]
    if len(holders) == 1:
        return QuoteCheck("", quote, "one chunk", holders)
    if holders:
        return QuoteCheck("", quote, "several chunks", holders)
    # Split across a boundary? The longest leading part sits in one chunk, the longest trailing part
    # in another, and together they cover the quote.
    words = target.split()
    texts = [(chunk_id, normal(text)) for chunk_id, text in chunks]

    def holder(part: list[str]) -> str | None:
        joined = " ".join(part)
        return next((chunk_id for chunk_id, text in texts if joined in text), None)

    head_end, head_chunk = next(((i, c) for i in range(len(words) - 1, 0, -1) if (c := holder(words[:i]))), (0, None))
    tail_start, tail_chunk = next(
        ((j, c) for j in range(1, len(words)) if (c := holder(words[j:]))), (len(words), None)
    )
    if head_chunk and tail_chunk and head_chunk != tail_chunk and tail_start <= head_end:
        return QuoteCheck("", quote, "split", sorted({head_chunk, tail_chunk}))
    return QuoteCheck("", quote, "missing", [])


def resolve_gold(
    conn: sqlite3.Connection, questions: list[EvalQuestion]
) -> tuple[dict[str, list[str]], list[QuoteCheck]]:
    """Gold chunk ids per answerable question, plus a check for every quote."""
    gold: dict[str, list[str]] = {}
    checks: list[QuoteCheck] = []
    cache: dict[str, list[tuple[str, str]]] = {}
    for question in questions:
        if not question.answerable:
            continue
        if question.doc_id not in cache:
            cache[question.doc_id] = [(c.chunk_id, c.text) for c in db.get_chunks(conn, question.doc_id)]
        chunk_ids: list[str] = []
        for quote in question.quotes:
            check = find_quote(quote, cache[question.doc_id])
            check.question_id = question.id
            checks.append(check)
            if check.status in ("one chunk", "several chunks"):
                chunk_ids.extend(c for c in check.chunk_ids if c not in chunk_ids)
        gold[question.id] = chunk_ids
    return gold, checks


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def word_overlap(question: str, chunk_text: str) -> float:
    """Share of the question's content words that appear in the chunk: high means the question
    copies the source's wording, which flatters lexical and dense search alike."""
    words = content_words(question)
    return len(words & content_words(chunk_text)) / len(words) if words else 0.0
