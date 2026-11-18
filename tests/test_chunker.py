import pytest

from rag.chunk.chunker import Chunker, sentence_spans
from rag.models import Section

DOC = "arxiv:2401.00001"


class WordCounter:
    """Stand-in tokenizer: one token per whitespace-separated word."""

    max_tokens = 512

    def count(self, text: str) -> int:
        return len(text.split())


def words(n: int, word: str = "w") -> str:
    return " ".join([word] * n)


def section(ord: int, path: str, paragraphs: list[str], pages: list[int] | None = None) -> Section:
    pages = pages or [1] * len(paragraphs)
    return Section.from_paragraphs(DOC, ord, path, 1, path, [(p, pg, pg) for p, pg in zip(paragraphs, pages)])


def chunker(max_tokens: int = 10) -> Chunker:
    return Chunker(WordCounter(), max_tokens)


def test_every_chunk_is_an_exact_slice_of_its_section():
    sections = [
        section(0, "1 Intro", ["Alpha beta gamma.", words(4, "x"), words(6, "y")]),
        section(1, "2 Method", [words(3, "z")]),
    ]
    chunks = chunker(8).chunk_document(sections)
    by_ord = {s.ord: s for s in sections}
    for chunk in chunks:
        assert by_ord[chunk.section_ord].text[chunk.char_start:chunk.char_end] == chunk.text
        assert chunk.n_tokens <= 8
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_paragraphs_are_packed_and_never_cross_sections():
    sections = [
        section(0, "1 Intro", [words(3, "a"), words(3, "b"), words(3, "c")]),
        section(1, "2 Method", [words(2, "d")]),
    ]
    chunks = chunker(7).chunk_document(sections)
    assert [(c.section_ord, c.text) for c in chunks] == [
        (0, "a a a\n\nb b b"),
        (0, "c c c"),
        (1, "d d"),  # short, but never merged with the previous section
    ]


def test_long_paragraph_splits_at_sentences_with_one_sentence_overlap():
    paragraph = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    [s] = [section(0, "1 Intro", [paragraph])]
    chunks = chunker(6).chunk_document([s])
    assert [c.text for c in chunks] == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",  # repeats the sentence before the boundary
        "Seven eight nine. Ten eleven twelve.",
    ]


def test_no_overlap_at_paragraph_boundaries():
    s = section(0, "1 Intro", [words(5, "a"), words(5, "b")])
    assert [c.text for c in chunker(6).chunk_document([s])] == [words(5, "a"), words(5, "b")]


def test_over_long_sentence_is_cut_at_words_and_long_words_at_characters():
    long_sentence = " ".join(f"t{i}" for i in range(9))
    s = section(0, "1 Intro", [long_sentence, "x" * 60])

    class CharCounter(WordCounter):
        def count(self, text):  # long "words" cost one token per 10 characters
            return sum(max(1, len(w) // 10) for w in text.split())

    chunks = Chunker(CharCounter(), 4).chunk_document([s])
    assert all(c.n_tokens <= 4 for c in chunks)
    assert [c.text for c in chunks][:3] == ["t0 t1 t2 t3", "t4 t5 t6 t7", "t8"]
    assert [len(c.text) for c in chunks[3:]] == [49, 11]  # 60 chars cut where 4 tokens stop fitting


def test_chunk_types():
    sections = [
        section(0, "Abstract", ["We propose a thing."]),
        section(1, "1 Intro", ["Body text."]),
        section(2, "Figures and Tables", ["Figure 1: A plot.", "Table 1: Numbers."]),
    ]
    chunks = chunker(50).chunk_document(sections)
    assert [(c.chunk_type, c.text) for c in chunks] == [
        ("abstract", "We propose a thing."),
        ("body", "Body text."),
        ("caption", "Figure 1: A plot."),  # one caption per chunk, even when both would fit
        ("caption", "Table 1: Numbers."),
    ]


def test_page_range_comes_from_overlapped_paragraphs():
    s = section(0, "1 Intro", [words(3, "a"), words(3, "b"), words(3, "c")], pages=[2, 3, 5])
    chunks = chunker(7).chunk_document([s])
    assert [(c.page_start, c.page_end) for c in chunks] == [(2, 3), (5, 5)]


def test_duplicate_text_in_a_section_is_kept_once():
    s = section(0, "1 Intro", ["Same text.", words(9, "z"), "Same text."])
    chunks = chunker(10).chunk_document([s])
    assert [c.text for c in chunks].count("Same text.") == 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_ids_are_stable_across_runs_and_record_the_version():
    s = section(0, "1 Intro", [words(20, "a")])
    first = chunker(8).chunk_document([s])
    second = chunker(8).chunk_document([s])
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunker_version == "v2-8"


def test_max_tokens_must_leave_room_for_the_header():
    with pytest.raises(ValueError, match="no room"):
        Chunker(WordCounter(), 500)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We use Adam. It works.", ["We use Adam.", "It works."]),
        ("As in Fig. 3 we see e.g. growth. Done.", ["As in Fig. 3 we see e.g. growth.", "Done."]),
        ("Kingma et al. 2014 propose it. Next.", ["Kingma et al. 2014 propose it.", "Next."]),
        ("Work by J. Smith shows it. Next.", ["Work by J. Smith shows it.", "Next."]),
        ("This is general. Next one.", ["This is general.", "Next one."]),
        ("Accuracy is 3.5 percent. (See below.) Fine.", ["Accuracy is 3.5 percent.", "(See below.)", "Fine."]),
    ],
)
def test_sentence_spans(text, expected):
    assert [text[a:b] for a, b in sentence_spans(text, 0, len(text))] == expected
