from pathlib import Path

import pytest
import requests

from rag.models import Section
from rag.parse import grobid as grobid_module
from rag.parse.base import ParsedDocument, ParseError, quality_issues
from rag.parse.grobid import GrobidParser
from rag.parse.tei import _is_junk_heading, tei_to_parsed

FIXTURE = Path(__file__).parent / "fixtures" / "sample.tei.xml"


@pytest.fixture(scope="module")
def paper() -> ParsedDocument:
    return tei_to_parsed(FIXTURE.read_text(), "arxiv:2401.00001", parser="grobid-test")


def paths(paper: ParsedDocument) -> list[str]:
    return [s.path for s in paper.sections]


def section(paper: ParsedDocument, path: str) -> Section:
    return next(s for s in paper.sections if s.path == path)


def test_header(paper):
    assert paper.title == "Tiny Adapters for Large Models"
    assert paper.authors == ["Ada Lovelace", "Alan M Turing"]
    # footnotes GROBID files under the abstract are dropped
    assert paper.abstract == "We freeze the base model and train small adapters."
    assert section(paper, "Abstract").text == "We freeze the base model and train small adapters."
    assert paper.n_pages == 4
    assert paper.n_references == 2


def test_section_tree(paper):
    assert paths(paper) == [
        "Abstract",
        "1 Introduction",
        "1 Introduction > Model",
        "3 Method",
        "3 Method > Double Quantization",
        "5.1 Setup",
        "5.2 Results",
        "6 Conclusion",
        "Appendix > B Proofs Left in the Body",
        "Appendix > A Extra Experiments",
        "Appendix > A Extra Experiments > Data Details",
        "Appendix > B.2 Hyperparameters",
        "Figures and Tables",
    ]
    levels = {s.path: s.level for s in paper.sections}
    assert levels["1 Introduction"] == 1
    assert levels["3 Method > Double Quantization"] == 2
    assert levels["5.1 Setup"] == 2
    assert levels["Appendix > A Extra Experiments"] == 2
    assert levels["Appendix > B.2 Hyperparameters"] == 3


def test_section_ords_are_consecutive(paper):
    assert [s.ord for s in paper.sections] == list(range(len(paper.sections)))


def test_inline_refs_and_formulas(paper):
    intro = section(paper, "1 Introduction")
    assert intro.text == "Adapters cut the number of trainable parameters (see Table 1)."
    method = section(paper, "3 Method")
    assert method.text == "We insert rank r matrices.\n\nW = W0 + BA (1)"


def test_junk_headings_continue_previous_section(paper):
    assert section(paper, "5.2 Results").text == (
        "Adapters match full finetuning.\n\nA chat line that GROBID read as a heading."
    )
    assert section(paper, "6 Conclusion").text == "Small is enough.\n\nAlgorithm box text."
    assert section(paper, "Appendix > B Proofs Left in the Body").text == (
        "Proof of the bound.\n\nFigure fragment."
    )


@pytest.mark.parametrize(
    ("title", "junk"),
    [
        ("Guanaco:", True), ("end for", True), ("(middle).", True), ("2.", True),
        ("If you have any other questions or if there's anything else I can help with today", True),
        ("Double Quantization", False), ("Model", False), ("PASCAL VOC", False),
        ("Label Smoothing", False), ("Skip-gram", False),
    ],
)
def test_junk_heading_rule(title, junk):
    assert _is_junk_heading(title) is junk


def test_headless_div_continues_previous_section(paper):
    dq = section(paper, "3 Method > Double Quantization")
    assert dq.text == "Quantize the constants too.\n\nA headless div continues the previous section."


def test_skipped_back_matter(paper):
    all_text = " ".join(s.text for s in paper.sections)
    assert "We thank everyone" not in all_text
    assert "Paper one" not in all_text


def test_captions(paper):
    captions = section(paper, "Figures and Tables")
    assert captions.text.split("\n\n") == [
        "Figure 1: Memory use of each method.",
        "Table 1: Trainable parameters per method.",
    ]


def test_paragraph_offsets_and_pages(paper):
    for s in paper.sections:
        joined = "\n\n".join(s.text[p.char_start:p.char_end] for p in s.paragraphs)
        assert joined == s.text
    model = section(paper, "1 Introduction > Model")
    assert (model.paragraphs[0].page_start, model.paragraphs[0].page_end) == (1, 2)
    assert section(paper, "Appendix > B.2 Hyperparameters").page_start == 4


def test_fixture_parse_is_suspect_only_for_length(paper):
    # The fixture is tiny, so the word-count gate fires; nothing else should.
    assert paper.status == "suspect"
    assert len(paper.issues) == 1 and "body words" in paper.issues[0]


def test_quality_gates_catch_garbled_and_empty_parses():
    garbled = Section.from_paragraphs("x", 1, "1 Intro", 1, "1 Intro", [("∂∑∫ ≈≠ 123 456 " * 100, 1, 1)])
    paper = ParsedDocument("x", "test", "", [], "", [garbled], 1, 0)
    issues = quality_issues(paper)
    assert "no title" in issues
    assert "no abstract" not in issues  # not a gate: most non-paper documents have no abstract
    assert any("body sections" in issue for issue in issues)
    assert any("letter ratio" in issue for issue in issues)


class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text
        self.headers = {"retry-after": "0"}


def test_grobid_retries_when_busy(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.5")
    responses = [FakeResponse(503), FakeResponse(200, "<TEI/>")]
    calls = []

    def fake_post(url, files, data, timeout):
        calls.append(dict(data).get("consolidateHeader"))
        return responses.pop(0)

    monkeypatch.setattr(grobid_module.requests, "post", fake_post)
    parser = GrobidParser("http://localhost:8070", timeout_seconds=10, max_retries=2)
    assert parser.process_pdf(pdf) == "<TEI/>"
    assert calls == ["0", "0"]  # consolidation is off on every attempt


def test_grobid_failure_raises_parse_error(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.5")
    monkeypatch.setattr(grobid_module.requests, "post", lambda *a, **k: FakeResponse(500, "boom"))
    with pytest.raises(ParseError, match="HTTP 500"):
        GrobidParser("http://localhost:8070", 10, 2).process_pdf(pdf)


def test_grobid_unreachable_raises_parse_error(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.5")

    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(grobid_module.requests, "post", refuse)
    monkeypatch.setattr(grobid_module.time, "sleep", lambda seconds: None)
    with pytest.raises(ParseError, match="not reachable"):
        GrobidParser("http://localhost:8070", 10, 2).process_pdf(pdf)
