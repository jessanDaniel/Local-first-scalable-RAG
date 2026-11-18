import pytest

from rag.answer.answerer import Answerer, ContextWindowError
from rag.answer.citations import check_citations, marker_numbers
from rag.answer.prompt import SYSTEM_PROMPT, build_user_prompt
from rag.answer.sources import Source, assemble_sources
from rag.llm.interface import LLM, Generation
from rag.models import Chunk, Document, Section
from rag.search import Hit
from rag.store import db

ATTN = "arxiv:1706.03762"
SECTION_0 = "We divide each dot product by sqrt dk. For large dk the dot products grow large in magnitude."
SECTION_1 = "Multi-head attention projects queries, keys and values h times."


class WordCounter:
    max_tokens = 100_000

    def count(self, text: str) -> int:
        return len(text.split())


def chunk(doc_id: str, section_ord: int, ord: int, section_text: str, text: str, page: int) -> Chunk:
    start = section_text.index(text)
    return Chunk(doc_id=doc_id, section_ord=section_ord, ord=ord, chunk_type="body", text=text,
                 char_start=start, char_end=start + len(text), n_tokens=len(text.split()),
                 chunker_version="test", page_start=page, page_end=page)


SENT_A = "We divide each dot product by sqrt dk."
SENT_B = "For large dk the dot products grow large in magnitude."
OVERLAP = SENT_A + " " + SENT_B


def texts(doc_id, section_ord):
    return {(ATTN, 0): SECTION_0, (ATTN, 1): SECTION_1}[(doc_id, section_ord)]


def hit(rank: int, c: Chunk, path: str = "3.2.1 Scaled Dot-Product Attention") -> Hit:
    return Hit(rank, 1.0 - rank / 100, c, "Attention Is All You Need", path, "https://arxiv.org/abs/1706.03762")


# --- citations ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "numbers"),
    [("2", [2]), ("1, 3", [1, 3]), ("2-4", [2, 3, 4]), ("2–3", [2, 3]), (" 5 ", [5])],
)
def test_marker_numbers(body, numbers):
    assert marker_numbers(body) == numbers


def test_check_citations():
    answer = "Scaling keeps softmax gradients usable [1]. It is cheap [1][2]. Heads differ. Also [7]."
    report = check_citations(answer, n_sources=2)
    assert report.cited == [1, 2]
    assert report.invalid == [7]
    assert (report.n_sentences, report.n_cited_sentences) == (4, 3)
    assert not check_citations("No sources were relevant.", 2).has_citations


# --- sources --------------------------------------------------------------------


def test_overlapping_hits_from_one_section_become_one_source():
    first = chunk(ATTN, 0, 0, SECTION_0, SENT_A, 4)
    overlapping = chunk(ATTN, 0, 1, SECTION_0, OVERLAP, 4)
    other = chunk(ATTN, 1, 2, SECTION_1, SECTION_1, 5)
    sources, dropped = assemble_sources(
        [hit(1, other, "3.2.2 Multi-Head Attention"), hit(2, overlapping), hit(3, first)],
        texts, WordCounter(), budget_tokens=1000, max_sources=8,
    )
    assert dropped == []
    assert [(s.n, s.section_path, s.best_rank) for s in sources] == [
        (1, "3.2.2 Multi-Head Attention", 1),
        (2, "3.2.1 Scaled Dot-Product Attention", 2),
    ]
    assert sources[1].text == SECTION_0  # merged span, read once
    assert sources[1].chunk_ids == [first.chunk_id, overlapping.chunk_id]


def test_budget_skips_a_source_that_does_not_fit_and_tries_the_next():
    big = chunk(ATTN, 0, 0, SECTION_0, SECTION_0, 4)
    small = chunk(ATTN, 1, 1, SECTION_1, SECTION_1, 5)
    budget = len(f"[1] Attention Is All You Need — x (p.5)\n{SECTION_1}".split()) + 6
    sources, dropped = assemble_sources([hit(1, big), hit(2, small, "x")], texts, WordCounter(), budget, 8)
    assert [s.section_path for s in sources] == ["x"] and sources[0].n == 1
    assert [h.chunk.chunk_id for h in dropped] == [big.chunk_id]


def test_max_sources():
    hits = [hit(1, chunk(ATTN, 0, 0, SECTION_0, SECTION_0, 4)), hit(2, chunk(ATTN, 1, 1, SECTION_1, SECTION_1, 5))]
    sources, dropped = assemble_sources(hits, texts, WordCounter(), 1000, max_sources=1)
    assert len(sources) == 1 and len(dropped) == 1


def test_source_label_and_links():
    source = Source(2, ATTN, "Attention Is All You Need", "3.2.1 Scaled", 4, 5, "t", ["c"], 1, 0.7, None)
    assert source.label == "[2] Attention Is All You Need (arxiv:1706.03762) §3.2.1 Scaled, pp.4-5"
    assert source.link == "https://arxiv.org/pdf/1706.03762#page=4"
    local = Source(1, "file:notes", "Notes", "Page 3", 3, 3, "t", ["c"], 1, 0.5, "/data/raw/files/file_notes.pdf")
    assert local.link == "file:///data/raw/files/file_notes.pdf#page=3"


def test_prompt_contains_numbered_sources_and_question():
    source = Source(1, ATTN, "Attention Is All You Need", "3.2.1 Scaled", 4, 4, "Divide by sqrt dk.", ["c"], 1, 0.7, None)
    prompt = build_user_prompt("Why scale?", [source])
    assert "[1] Attention Is All You Need — 3.2.1 Scaled (p.4)\nDivide by sqrt dk." in prompt
    assert prompt.endswith("Question: Why scale?")
    assert "[2]" in SYSTEM_PROMPT and "do not cover" in SYSTEM_PROMPT


# --- answerer -------------------------------------------------------------------


class FakeSearcher:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k):
        return self.hits[:k]


class FakeLLM(LLM):
    provider = "fake"
    model = "fake-1"

    def __init__(self, text: str, context_window: int = 8192, max_tokens: int = 100, completion_tokens: int = 20):
        self.text = text
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.completion_tokens = completion_tokens
        self.calls = []

    def generate(self, prompt, system=None):
        self.calls.append((system, prompt))
        return Generation(self.text, self.provider, self.model, 0.1, 50, self.completion_tokens)


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "rag.db")
    db.upsert_document(connection, Document(doc_id=ATTN, source="arxiv", title="Attention Is All You Need"))
    db.replace_sections(connection, ATTN, [
        Section.from_paragraphs(ATTN, 0, "3.2.1", 3, "3.2.1 Scaled Dot-Product Attention", [(SECTION_0, 4, 4)]),
        Section.from_paragraphs(ATTN, 1, "3.2.2", 3, "3.2.2 Multi-Head Attention", [(SECTION_1, 5, 5)]),
    ])
    yield connection
    connection.close()


def answerer(conn, llm, context_tokens=1000):
    hits = [hit(1, chunk(ATTN, 0, 0, SECTION_0, SECTION_0, 4))]
    return Answerer(conn, FakeSearcher(hits), llm, WordCounter(), 10, 8, context_tokens)


def test_answer_with_valid_citation(conn):
    llm = FakeLLM("Large dot products push softmax into small gradients, so they are scaled [1].")
    answer = answerer(conn, llm).ask("Why scale?")
    assert answer.citations.cited == [1] and answer.warnings == []
    assert [s.label for s in answer.cited_sources] == [
        "[1] Attention Is All You Need (arxiv:1706.03762) §3.2.1 Scaled Dot-Product Attention, p.4"
    ]
    system, prompt = llm.calls[0]
    assert system == SYSTEM_PROMPT and "Question: Why scale?" in prompt


def test_invented_and_missing_citations_are_flagged(conn):
    invented = answerer(conn, FakeLLM("It is scaled [3].")).ask("Why scale?")
    assert invented.citations.invalid == [3]
    assert any("do not exist" in w for w in invented.warnings)
    uncited = answerer(conn, FakeLLM("It is scaled.")).ask("Why scale?")
    assert "answer has no valid citations" in uncited.warnings


def test_answer_at_the_length_limit_is_flagged(conn):
    answer = answerer(conn, FakeLLM("Cut off [1]", max_tokens=100, completion_tokens=100)).ask("q")
    assert any("may be cut off" in w for w in answer.warnings)


def test_prompt_that_would_not_fit_is_refused_before_calling_the_model(conn):
    llm = FakeLLM("never sent", context_window=120, max_tokens=60)
    with pytest.raises(ContextWindowError, match="silently lose the start"):
        answerer(conn, llm).ask("Why scale?")
    assert llm.calls == []
