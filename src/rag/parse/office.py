"""Word (.docx) and HTML -> ParsedDocument, with the standard library only (no new dependency).

Word: a .docx file is a zip of XML. word/document.xml holds the paragraphs; a paragraph's style
("Heading 1", "Heading2", "Title") says whether it is a heading and at which level, so the section
tree comes from the author, not a guess. List items keep a "- " marker; a table becomes one
paragraph per row, cells joined with " | ". The title is docProps/core.xml's dc:title, else the
first "Title"-style paragraph. Word files have no fixed pages, so pages are unknown (None).

HTML: h1-h6 are headings; p, li, pre, blockquote, table rows and divs holding text are paragraphs.
script, style, nav, header, footer, aside, form and noscript are skipped (page furniture, not
content). The title is <title>, else the first h1. Pages: none.

Both refuse files whose XML / HTML is unreasonably large (a zip that expands to more than 50 MB of
document.xml), so a hostile file can't exhaust memory.
"""

import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from rag.parse.base import ParsedDocument, ParseError, Parser, quality_issues
from rag.parse.text import _SectionBuilder

MAX_XML_BYTES = 50_000_000
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.I)


def _finish(doc_id: str, parser: str, title: str, builder: _SectionBuilder, name: str) -> ParsedDocument:
    sections = builder.finish()
    if not sections:
        raise ParseError(f"no text found in {name}")
    parsed = ParsedDocument(
        doc_id=doc_id,
        parser=parser,
        title=title,
        authors=[],
        abstract="",
        sections=sections,
        n_pages=None,
        n_references=0,
    )
    parsed.issues = quality_issues(parsed)
    return parsed


# --- Word -----------------------------------------------------------------------------------------


def _read_xml(archive: zipfile.ZipFile, name: str) -> ElementTree.Element | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_XML_BYTES:
        raise ParseError(f"{name} expands to {info.file_size} bytes; refusing to parse")
    return ElementTree.fromstring(archive.read(name))


def _style_names(archive: zipfile.ZipFile) -> dict[str, str]:
    """styleId -> name ("Heading1" -> "heading 1"); localised Word versions use ids like "berschrift1"."""
    root = _read_xml(archive, "word/styles.xml")
    if root is None:
        return {}
    names = {}
    for style in root.iter(f"{W}style"):
        name = style.find(f"{W}name")
        if name is not None:
            names[style.get(f"{W}styleId", "")] = name.get(f"{W}val", "")
    return names


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts = []
    for node in paragraph.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag in (f"{W}tab",):
            parts.append(" ")
        elif node.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


class DocxParser(Parser):
    name = "docx-v1"

    def parse(self, path: Path, doc_id: str) -> ParsedDocument:
        try:
            with zipfile.ZipFile(path) as archive:
                body = _read_xml(archive, "word/document.xml")
                styles = _style_names(archive)
                core = _read_xml(archive, "docProps/core.xml")
        except (zipfile.BadZipFile, ElementTree.ParseError, OSError) as exc:
            raise ParseError(f"cannot read {path.name} as a Word document: {exc}") from exc
        if body is None:
            raise ParseError(f"{path.name} has no word/document.xml (not a Word document?)")

        title = ""
        if core is not None:
            node = core.find("{http://purl.org/dc/elements/1.1/}title")
            title = " ".join((node.text or "").split()) if node is not None else ""

        builder = _SectionBuilder(doc_id)
        container = body.find(f"{W}body")
        for element in list(container) if container is not None else []:
            if element.tag == f"{W}tbl":
                for row in element.iter(f"{W}tr"):
                    cells = [
                        " ".join(_paragraph_text(p) for p in cell.iter(f"{W}p")).strip() for cell in row.iter(f"{W}tc")
                    ]
                    if any(cells):
                        builder.paragraph(" | ".join(cells))
                continue
            if element.tag != f"{W}p":
                continue
            text = _paragraph_text(element)
            if not text:
                continue
            style_id = ""
            props = element.find(f"{W}pPr")
            if props is not None:
                style = props.find(f"{W}pStyle")
                style_id = style.get(f"{W}val", "") if style is not None else ""
            style_name = styles.get(style_id, style_id)
            heading = HEADING_STYLE.match(style_name.replace("Heading", "heading ").strip())
            if style_name.lower() == "title":
                title = title or text
            elif heading:
                builder.heading(text, int(heading.group(1)))
            elif props is not None and props.find(f"{W}numPr") is not None:
                builder.paragraph(f"- {text}")
            else:
                builder.paragraph(text)
        return _finish(doc_id, self.name, title, builder, path.name)


# --- HTML -----------------------------------------------------------------------------------------

SKIP = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "template", "svg"}
BLOCK = {"p", "li", "pre", "blockquote", "tr", "div", "section", "article", "main", "dd", "dt", "figcaption", "br"}
HEADINGS = {f"h{i}": i for i in range(1, 7)}


class _HTMLCollector(HTMLParser):
    def __init__(self, builder: _SectionBuilder, h1_is_title: bool):
        super().__init__(convert_charrefs=True)
        self.builder = builder
        self.h1_is_title = h1_is_title  # one <h1> on the page names it: title, and h2 becomes level 1
        self.skip_depth = 0
        self.text: list[str] = []
        self.heading: int | None = None
        self.title = ""
        self.in_title = False
        self.first_h1 = ""
        self.list_item = False
        self.cells: list[str] | None = None

    def flush(self) -> None:
        text = " ".join("".join(self.text).split())
        self.text = []
        if not text:
            return
        if self.heading is not None:
            if self.heading == 1 and not self.first_h1:
                self.first_h1 = text
                if self.h1_is_title:
                    return
            self.builder.heading(text, max(1, self.heading - 1) if self.h1_is_title else self.heading)
        else:
            self.builder.paragraph(f"- {text}" if self.list_item else text)

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        elif tag in HEADINGS:
            self.flush()
            self.heading = HEADINGS[tag]
        elif tag == "tr":
            self.flush()
            self.cells = []
        elif tag in ("td", "th") and self.cells is not None:
            self.text = []
        elif tag in BLOCK:
            self.flush()
            self.list_item = tag == "li"

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        elif tag in HEADINGS:
            self.flush()
            self.heading = None
        elif tag in ("td", "th") and self.cells is not None:
            self.cells.append(" ".join("".join(self.text).split()))
            self.text = []
        elif tag == "tr" and self.cells is not None:
            if any(self.cells):
                self.builder.paragraph(" | ".join(self.cells))
            self.cells = None
        elif tag in BLOCK:
            self.flush()
            self.list_item = False

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_title:
            self.title += data
        else:
            self.text.append(data)


class HTMLDocParser(Parser):
    name = "html-v1"

    def parse(self, path: Path, doc_id: str) -> ParsedDocument:
        if path.stat().st_size > MAX_XML_BYTES:
            raise ParseError(f"{path.name} is larger than {MAX_XML_BYTES} bytes; refusing to parse")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        builder = _SectionBuilder(doc_id)
        collector = _HTMLCollector(builder, h1_is_title=len(re.findall(r"<h1[\s>]", text, re.I)) == 1)
        collector.feed(text)
        collector.close()
        collector.flush()
        title = " ".join(collector.title.split()) or collector.first_h1
        return _finish(doc_id, self.name, title, builder, path.name)
