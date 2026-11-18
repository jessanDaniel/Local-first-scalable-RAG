from datetime import date
from pathlib import Path

import pytest

from rag.acquire.arxiv import ArxivError, parse_feed, read_id_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_feed():
    adam, old = parse_feed((FIXTURES / "arxiv_feed.xml").read_text())
    assert adam.arxiv_id == "1412.6980"
    assert adam.version == 9
    assert adam.title == "Adam: A Method for Stochastic Optimization"
    assert adam.authors == ["Diederik P. Kingma", "Jimmy Ba"]
    assert adam.abstract == "We introduce Adam, an algorithm for first-order gradient-based optimization."
    assert adam.categories == ["cs.LG", "cs.AI"]
    assert adam.primary_category == "cs.LG"
    assert (adam.published, adam.updated) == (date(2014, 12, 22), date(2017, 1, 30))
    assert adam.source_format == "pdf"
    assert adam.parse_status == "pending"
    assert (old.arxiv_id, old.version) == ("hep-th/9901001", 2)


def test_error_entry_raises():
    feed = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id>
      <title>Error</title><summary>incorrect id format for 1234</summary></entry></feed>"""
    with pytest.raises(ArxivError, match="incorrect id format"):
        parse_feed(feed)


def test_read_id_file(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# header\n\n1412.6980   # Adam\n2006.11239\nhep-th/9901001\n")
    assert read_id_file(path) == ["1412.6980", "2006.11239", "hep-th/9901001"]


@pytest.mark.parametrize("bad", ["1412.6980v9", "not-an-id", "https://arxiv.org/abs/1412.6980"])
def test_read_id_file_rejects_bad_ids(tmp_path, bad):
    path = tmp_path / "ids.txt"
    path.write_text(f"{bad}\n")
    with pytest.raises(ValueError, match="not an arXiv id"):
        read_id_file(path)


def test_read_id_file_rejects_duplicates(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("1412.6980\n1412.6980\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_id_file(path)


def test_seed_corpus_file_is_valid():
    ids = read_id_file(Path(__file__).parents[1] / "corpus" / "seed_ai_research.txt")
    assert len(ids) == 10
