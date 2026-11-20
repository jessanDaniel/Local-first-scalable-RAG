import pytest

from rag.index.tokenize import query_terms, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BLEU score of 34.81", {"bleu", "score", "34", "81", "34.81"}),
        ("GPT-4 and ResNet-50", {"gpt", "4", "gpt-4", "resnet", "50", "resnet-50"}),
        ("ResNet-50/101/152", {"resnet-50/101/152", "101", "152"}),
        ("β2 = 0.999", {"β2", "β", "2", "0", "999", "0.999"}),
        ("β 2", {"β", "2"}),  # how PDF extraction often prints β2
        ("resnet50", {"resnet50", "resnet", "50"}),
        ("Eﬃcient", {"efficient"}),  # ligature expanded
        ("3.57% top-5 error", {"3.57", "3", "57", "top-5", "top", "5", "error"}),
    ],
)
def test_tokens_cover_whole_terms_and_their_parts(text, expected):
    assert expected <= set(tokenize(text))


def test_no_stemming_yet():
    assert "refund" not in tokenize("refunds")


def test_stopwords_dropped_numbers_kept():
    tokens = tokenize("What is the size of the 2 models?")
    assert "the" not in tokens and "what" not in tokens and "2" in tokens


def test_term_frequency_is_kept_but_query_terms_are_distinct():
    assert tokenize("loss loss loss").count("loss") == 3
    assert query_terms("loss loss") == ["loss"]


def test_queries_and_documents_tokenize_the_same_way():
    assert set(query_terms("What BLEU score did the LSTM reach?")) - {"reach"} <= set(
        tokenize("The LSTM had a BLEU score of 34.8")
    )
