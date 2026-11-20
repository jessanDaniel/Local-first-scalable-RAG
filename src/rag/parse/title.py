"""Choosing a document title when the source does not give one (local files).

A parser's title is a guess. GROBID, for instance, took the arXiv licence line
("Provided proper attribution is provided, Google hereby grants ...") and a venue
banner ("Published as a conference paper at ICLR 2015 ADAM: ...") as titles. The
title goes into every chunk's embedding header and every citation, so it is
checked, not trusted: candidates are tried in order and the first one that passes
`title_problem` wins. A title that did not come from the first candidate is
reported, so the document can be reviewed.
"""

import re
import unicodedata

MAX_TITLE_WORDS = 30
MAX_TITLE_CHARS = 250
MIN_LETTER_RATIO = 0.5

# Text that sits where a title would but is not one: notices, venue banners, stamps,
# placeholder metadata from authoring tools.
NOT_A_TITLE = [
    (
        # Notice phrasing, not single words: "Permission-Based Access Control" is a real title.
        re.compile(
            r"\b(hereby grants?|permission to (make|reproduce|copy|use)|proper attribution|copyright \d{4}"
            r"|all rights reserved|licensed under)\b|©",
            re.I,
        ),
        "licence or copyright notice",
    ),
    (re.compile(r"^(published|accepted|presented|appearing|to appear)\b", re.I), "venue banner"),
    (re.compile(r"\b(under review|preprint)\b", re.I), "review or preprint banner"),
    (re.compile(r"arxiv:\s*\d", re.I), "arXiv stamp"),
    (re.compile(r"^proceedings of\b", re.I), "proceedings header"),
    (re.compile(r"@|https?://|www\.", re.I), "contains an email or URL"),
    (re.compile(r"^(microsoft word\b|untitled\b)|\.(docx?|tex|pdf)$", re.I), "authoring-tool placeholder"),
]


def clean_title(text: str) -> str:
    """NFKC (the "ﬁ" ligature becomes "fi"), single spaces."""
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def title_problem(text: str) -> str | None:
    """Why this is not a usable title, or None if it is."""
    title = clean_title(text)
    letters = [c for c in title if not c.isspace()]
    if sum(c.isalnum() for c in letters) < 2:
        return "empty"
    if len(title) > MAX_TITLE_CHARS or len(title.split()) > MAX_TITLE_WORDS:
        return "too long"
    for pattern, reason in NOT_A_TITLE:  # before the letter ratio: the specific reason is the useful one
        if pattern.search(title):
            return reason
    if sum(c.isalpha() for c in letters) / len(letters) < MIN_LETTER_RATIO:
        return "mostly not letters"
    return None


def choose_title(candidates: list[tuple[str, str]]) -> tuple[str, str, list[str]]:
    """First usable (source, text) candidate -> (title, source, notes on rejected candidates).

    The last candidate is the fallback (the file name) and is used even if it fails the checks.
    """
    rejected = []
    for source, text in candidates:
        problem = title_problem(text)
        if problem is None:
            return clean_title(text), source, rejected
        if clean_title(text):
            rejected.append(f"{source} title rejected ({problem}): {clean_title(text)[:60]}")
    source, text = candidates[-1]
    return clean_title(text), source, rejected
