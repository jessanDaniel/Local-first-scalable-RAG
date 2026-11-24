"""GROBID TEI XML -> ParsedDocument. No network: works on a saved TEI file.

The rules come from parsing QLoRA (arXiv 2305.14314) by hand:

- Numbered headings ("3", "5.1") form the section tree. A parent that GROBID
  did not emit (5.1 with no "5") is simply left out of the path.
- An unnumbered heading becomes a child of the last numbered section
  ("3 QLORA Finetuning > Double Quantization"), never top-level.
- A heading GROBID made out of other text ("Guanaco:" from a chat transcript,
  "end for" from an algorithm box, "(middle)." from a figure) is dropped and its
  text continues the previous section. Same for a div with no heading.
- Appendix headings carry their number in the text ("B.2 Hyperparameters");
  their paths start with "Appendix". GROBID sometimes leaves the appendix in
  <body>: a lettered heading ("A Extended derivations") after at least two
  numbered sections switches the rest of the body to appendix paths.
  (Checked by hand on the first 10 papers.)
- Acknowledgements, funding and references are not body text.
- A figure or table counts as a caption only if it is labelled "Figure N" or
  "Table N"; the rest (e.g. a chat example GROBID boxed as a figure) is dropped.
- Pages come from each element's coords ("page,x,y,w,h;...").
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from rag.models import Section
from rag.parse.base import (
    ABSTRACT_PATH,
    APPENDIX_PATH,
    CAPTIONS_PATH,
    ParsedDocument,
    quality_issues,
)

TEI = "{http://www.tei-c.org/ns/1.0}"
SKIPPED_DIV_TYPES = {"acknowledgement", "funding", "availability", "references", "toc"}
PATH_SEPARATOR = " > "

BODY_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
APPENDIX_NUMBER = re.compile(r"^([A-Z](?:\.\d+)*|\d+(?:\.\d+)*)\.?\s+(\S.*)$")
CAPTION_LABEL = re.compile(r"^(fig\.?|figure|table)\s*\d+", re.IGNORECASE)
# "A Extended derivations", "B.2 Hyperparameters", "C. ImageNet Localization"
LETTERED_HEADING = re.compile(r"^[A-Z](?:\.\d+)*\.?\s+[A-Z0-9]")
MIN_SECTIONS_BEFORE_BODY_APPENDIX = 2
FOOTNOTE_MARK = re.compile(r"^[*†‡§¶‖]")
JUNK_HEADING_MAX_WORDS = 12


def tei_to_parsed(tei_xml: str, doc_id: str, parser: str) -> ParsedDocument:
    root = ET.fromstring(tei_xml)
    header = root.find(f"{TEI}teiHeader")
    body = root.find(f"{TEI}text/{TEI}body")
    back = root.find(f"{TEI}text/{TEI}back")

    abstract_paragraphs = _abstract_paragraphs(header)
    drafts: list[_Draft] = []
    if abstract_paragraphs:
        drafts.append(_Draft(ABSTRACT_PATH, 1, ABSTRACT_PATH, abstract_paragraphs))

    if body is not None:
        builder = _SectionBuilder(prefix=None, number_pattern=BODY_NUMBER)
        builder.collect(body)
        drafts.extend(builder.drafts)
    if back is not None:
        for annex in back.findall(f"{TEI}div[@type='annex']"):
            builder = _SectionBuilder(prefix=APPENDIX_PATH, number_pattern=APPENDIX_NUMBER)
            builder.collect(annex)
            drafts.extend(builder.drafts)

    captions = _captions(root)
    if captions:
        drafts.append(_Draft(CAPTIONS_PATH, 1, CAPTIONS_PATH, captions))

    sections = [
        Section.from_paragraphs(doc_id, ord, d.heading, d.level, d.path, d.paragraphs)
        for ord, d in enumerate(d for d in drafts if d.paragraphs)
    ]
    parsed = ParsedDocument(
        doc_id=doc_id,
        parser=parser,
        title=_title(header),
        authors=_authors(header),
        abstract=" ".join(text for text, _, _ in abstract_paragraphs),
        sections=sections,
        n_pages=_page_count(root),
        n_references=len(root.findall(f".//{TEI}back//{TEI}listBibl/{TEI}biblStruct")),
    )
    parsed.issues = quality_issues(parsed)
    return parsed


# --- text helpers -----------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _inline_text(element: ET.Element | None) -> str:
    """Text of an element whose children are inline (refs inside a paragraph)."""
    return _normalize("".join(element.itertext())) if element is not None else ""


def _spaced_text(element: ET.Element | None) -> str:
    """Text of an element whose children are separate words (name parts, formula labels)."""
    return _normalize(" ".join(element.itertext())) if element is not None else ""


def _pages(element: ET.Element) -> tuple[int | None, int | None]:
    coords = element.get("coords")
    if not coords:
        return None, None
    pages = [int(box.split(",", 1)[0]) for box in coords.split(";") if box.strip()]
    return min(pages), max(pages)


# --- header -----------------------------------------------------------------


def _title(header: ET.Element) -> str:
    title = header.find(f"{TEI}fileDesc/{TEI}titleStmt/{TEI}title[@type='main']")
    if title is None:
        title = header.find(f"{TEI}fileDesc/{TEI}titleStmt/{TEI}title")
    return _inline_text(title)


def _authors(header: ET.Element) -> list[str]:
    names = []
    for person in header.findall(
        f"{TEI}fileDesc/{TEI}sourceDesc/{TEI}biblStruct/{TEI}analytic/{TEI}author/{TEI}persName"
    ):
        name = _spaced_text(person)
        if name:
            names.append(name)
    return names


def _abstract_paragraphs(header: ET.Element) -> list[tuple[str, int | None, int | None]]:
    abstract = header.find(f"{TEI}profileDesc/{TEI}abstract")
    if abstract is None:
        return []
    paragraphs = []
    for p in abstract.iter(f"{TEI}p"):
        text = _inline_text(p)
        # GROBID sometimes files first-page footnotes under the abstract
        # ("* Equal contribution...", "† Work performed while at ...").
        if text and not FOOTNOTE_MARK.match(text):
            paragraphs.append((text, *_pages(p)))
    return paragraphs


def _page_count(root: ET.Element) -> int | None:
    facsimile = root.find(f"{TEI}facsimile")
    if facsimile is not None and len(facsimile):
        return len(facsimile.findall(f"{TEI}surface"))
    pages = [page for el in root.iter() for page in _pages(el) if page is not None]
    return max(pages) if pages else None


# --- sections ---------------------------------------------------------------


@dataclass
class _Draft:
    heading: str
    level: int
    path: str
    paragraphs: list[tuple[str, int | None, int | None]] = field(default_factory=list)


def _is_junk_heading(title: str) -> bool:
    """A heading GROBID made out of non-heading text: algorithm lines ("end for"),
    speaker labels ("Guanaco:"), figure fragments ("(middle)."), stray numbers ("2."),
    or a whole sentence."""
    words = title.split()
    return (
        not title[0].isalnum()
        or title[0].islower()
        or title.endswith((":", "."))
        or len(words) > JUNK_HEADING_MAX_WORDS
        or not any(sum(c.isalpha() for c in word) >= 3 for word in words)
    )


class _SectionBuilder:
    def __init__(self, prefix: str | None, number_pattern: re.Pattern):
        self.prefix = prefix
        self.number_pattern = number_pattern
        self.numbered: list[tuple[str, int, str]] = []  # open numbered ancestors: (number, level, label)
        self.drafts: list[_Draft] = []
        self.top_level_sections = 0

    def collect(self, container: ET.Element) -> None:
        for div in container.findall(f"{TEI}div"):
            if div.get("type") in SKIPPED_DIV_TYPES:
                continue
            head = div.find(f"{TEI}head")
            content = [el for el in div if el.tag in (f"{TEI}p", f"{TEI}formula")]
            if head is not None or content:
                draft = self._open(head)
                for el in content:
                    text = _inline_text(el) if el.tag == f"{TEI}p" else _spaced_text(el)
                    if text:
                        draft.paragraphs.append((text, *_pages(el)))
            self.collect(div)

    def _open(self, head: ET.Element | None) -> _Draft:
        title = _inline_text(head)
        if not title:
            if self.drafts:
                return self.drafts[-1]
            title = "Untitled"

        number = head.get("n").rstrip(".") if head is not None and head.get("n") else None
        if number is None and self._starts_appendix_in_body(title):
            # GROBID sometimes leaves the appendix in <body>: "A Extended derivations",
            # "E Monte Carlo EM". From here on, this builder is producing appendix sections.
            self.prefix = APPENDIX_PATH
            self.number_pattern = APPENDIX_NUMBER
            self.numbered = []
        if number is None:
            match = self.number_pattern.match(title)
            if match:
                number, title = match.group(1), match.group(2)
        if number is None and _is_junk_heading(title) and self.drafts:
            return self.drafts[-1]

        if number is not None:
            level = number.count(".") + 1
            if level == 1:
                self.top_level_sections += 1
            self.numbered = [
                (n, lvl, label) for n, lvl, label in self.numbered if lvl < level and number.startswith(n + ".")
            ]
            label = f"{number} {title}"
            parts = [label for _, _, label in self.numbered] + [label]
            self.numbered.append((number, level, label))
        else:
            label = title
            level = self.numbered[-1][1] + 1 if self.numbered else 1
            parts = [label for _, _, label in self.numbered] + [label]

        if self.prefix:
            parts = [self.prefix, *parts]
            level += 1
        draft = _Draft(heading=label, level=level, path=PATH_SEPARATOR.join(parts))
        self.drafts.append(draft)
        return draft

    def _starts_appendix_in_body(self, title: str) -> bool:
        return (
            self.prefix is None
            and self.top_level_sections >= MIN_SECTIONS_BEFORE_BODY_APPENDIX
            and LETTERED_HEADING.match(title) is not None
        )


# --- captions ---------------------------------------------------------------


def _captions(root: ET.Element) -> list[tuple[str, int | None, int | None]]:
    captions = []
    seen = set()
    for figure in root.iter(f"{TEI}figure"):
        description = _inline_text(figure.find(f"{TEI}figDesc"))
        label = _inline_text(figure.find(f"{TEI}head")).rstrip(" :.")
        if not description:
            continue
        if CAPTION_LABEL.match(description):
            text = description
        elif CAPTION_LABEL.match(label):
            text = f"{label}: {description}"
        else:
            continue
        if text not in seen:
            seen.add(text)
            captions.append((text, *_pages(figure)))
    return captions
