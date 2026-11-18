"""arXiv metadata and PDF downloads.

Metadata comes from the arXiv API (Atom XML) and is the authoritative source
for title, authors, abstract, categories, dates and version; the PDF parser
only supplies the body. Every request waits arxiv_delay_seconds after the
previous one, as arXiv asks, and sends a generic User-Agent.
"""

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import requests

from rag.config import Settings, get_settings
from rag.llm.http import backoff_seconds
from rag.models import Document, make_doc_id

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ID_WITH_VERSION = re.compile(r"abs/(?P<id>[^v/]+(?:/[^v]+)?)v(?P<version>\d+)$")
ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})$")
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 60.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class ArxivError(RuntimeError):
    """arXiv could not be reached, or returned something unusable."""


def read_id_file(path: Path) -> list[str]:
    """One arXiv id per line; blank lines and anything after '#' are ignored."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        arxiv_id = line.split("#", 1)[0].strip()
        if not arxiv_id:
            continue
        if not ARXIV_ID.match(arxiv_id):
            raise ValueError(f"{path}: not an arXiv id without version: {arxiv_id!r}")
        ids.append(arxiv_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return ids


def parse_feed(atom_xml: str) -> list[Document]:
    """arXiv API Atom feed -> Documents (parse_status 'pending', nothing parsed yet).

    doc_id is "arxiv:<id>"; arXiv-only fields (id, version, categories, last update)
    go in metadata."""
    root = ET.fromstring(atom_xml)
    documents = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = (entry.findtext(f"{ATOM}id") or "").strip()
        match = ID_WITH_VERSION.search(entry_id)
        if not match:
            # The API answers an unknown id with an entry whose id is the API error page.
            raise ArxivError(f"arXiv returned an error entry: {_text(entry, 'summary') or entry_id}")
        arxiv_id = match.group("id")
        primary = entry.find(f"{ARXIV}primary_category")
        updated = _date(entry.findtext(f"{ATOM}updated"))
        documents.append(
            Document(
                doc_id=make_doc_id("arxiv", arxiv_id),
                source="arxiv",
                title=_text(entry, "title"),
                authors=[_text(author, "name") for author in entry.findall(f"{ATOM}author")],
                abstract=_text(entry, "summary"),
                published=_date(entry.findtext(f"{ATOM}published")),
                source_uri=f"https://arxiv.org/abs/{arxiv_id}",
                source_format="pdf",
                metadata={
                    "arxiv_id": arxiv_id,
                    "version": int(match.group("version")),
                    "categories": [c.get("term") for c in entry.findall(f"{ATOM}category")],
                    "primary_category": primary.get("term") if primary is not None else "",
                    "updated": updated.isoformat() if updated else None,
                },
            )
        )
    return documents


def _text(element: ET.Element, tag: str) -> str:
    return " ".join((element.findtext(f"{ATOM}{tag}") or "").split())


def _date(value: str | None) -> date | None:
    return datetime.fromisoformat(value.strip()).date() if value else None


class ArxivClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.settings.http_user_agent
        self._last_request = 0.0

    def fetch_metadata(self, arxiv_ids: list[str]) -> list[Document]:
        """Documents for the given arXiv ids, in the same order. Fails if any id is unknown."""
        documents: list[Document] = []
        for start in range(0, len(arxiv_ids), 100):
            batch = arxiv_ids[start : start + 100]
            response = self._get(
                self.settings.arxiv_api_url,
                params={"id_list": ",".join(batch), "max_results": len(batch)},
            )
            documents.extend(parse_feed(response.text))
        by_id = {document.metadata["arxiv_id"]: document for document in documents}
        missing = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in by_id]
        if missing:
            raise ArxivError(f"arXiv returned no metadata for: {', '.join(missing)}")
        return [by_id[arxiv_id] for arxiv_id in arxiv_ids]

    def download_pdf(self, document: Document, pdf_dir: Path) -> tuple[Path, str, bool]:
        """Save the document's PDF as <pdf_dir>/<arxiv id>.pdf.

        Returns (path, sha256, downloaded). An existing file is reused, not fetched again.
        """
        arxiv_id = document.metadata["arxiv_id"]
        path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        if path.exists() and path.stat().st_size > 0:
            return path, _sha256(path.read_bytes()), False

        url = f"{self.settings.arxiv_pdf_url}/{arxiv_id}v{document.metadata['version']}"
        response = self._get(url)
        if not response.content.startswith(b"%PDF"):
            raise ArxivError(f"{url} did not return a PDF (content-type {response.headers.get('content-type')})")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".pdf.part")
        partial.write_bytes(response.content)
        partial.rename(path)  # never leave a half-written PDF under the final name
        return path, _sha256(response.content), True

    def _get(self, url: str, params: dict | None = None, max_retries: int = 3) -> requests.Response:
        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(
                    url, params=params, timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
                )
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise ArxivError(f"GET {url} failed: {exc}") from exc
                time.sleep(backoff_seconds(attempt, None))
                continue
            if response.status_code in RETRY_STATUS_CODES and attempt < max_retries:
                time.sleep(backoff_seconds(attempt, response.headers.get("retry-after")))
                continue
            if not response.ok:
                raise ArxivError(f"GET {url}: HTTP {response.status_code}")
            return response
        raise AssertionError("unreachable: the last attempt always returns or raises")

    def _throttle(self) -> None:
        wait = self._last_request + self.settings.arxiv_delay_seconds - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
