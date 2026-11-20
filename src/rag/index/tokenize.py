"""BM25 tokenizer: text -> index terms. Used for chunks and queries alike, so they always agree.

Research text is full of tokens a plain word splitter breaks: "GPT-4", "34.81",
"ResNet-50", "β2" (which PDF extraction often prints as "β 2"). So each piece
of text yields:

    words       unicode letter/digit runs, lowercased:          "gpt", "4", "34", "81"
    compounds   words joined by . - / with no space between:   "gpt-4", "34.81", "50/101/152"
    splits      letter/digit boundaries inside a word:          "resnet50" -> "resnet", "50"

Duplicates are kept, so term frequency still counts. Common English stopwords
are dropped; numbers never are. No stemming yet ("refunds" does not match
"refund"): the eval decides whether that costs anything before we add it.
"""

import re
import unicodedata

TOKENIZER_VERSION = "t1"

WORD = re.compile(r"[^\W_]+")  # letters and digits, any script
COMPOUND = re.compile(r"[^\W_]+(?:[.\-/][^\W_]+)+")  # words joined by . - / with no space
LETTER_DIGIT = re.compile(r"[^\W\d_]+|\d+")  # a run of letters, or a run of digits
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it its "
    "of on or our so such than that the their them then there these they this those to was we were what when "
    "where which while who why will with would you your".split()
)


def normalise(text: str) -> str:
    """NFKC (ligatures like "ﬁ" become "fi", full-width digits become ASCII), lowercase."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    text = normalise(text)
    tokens: list[str] = []
    for word in WORD.findall(text):
        if word in STOPWORDS:
            continue
        tokens.append(word)
        parts = LETTER_DIGIT.findall(word)
        if len(parts) > 1:
            tokens.extend(parts)
    tokens.extend(COMPOUND.findall(text))
    return tokens


def query_terms(text: str) -> list[str]:
    """Distinct terms of a query, in first-seen order (each is scored once)."""
    return list(dict.fromkeys(tokenize(text)))
