"""GROBID implementation of the Parser interface.

GROBID runs as a local Java service (see AGENTS.md, environment notes). This
module only talks HTTP to it; turning its TEI XML into sections lives in
rag.parse.tei so it can be tested without a running service.

Consolidation is always off: with it on, GROBID can look papers up on
CrossRef, which would send data off this machine.
"""

import time
from pathlib import Path

import requests

from rag.config import Settings, get_settings
from rag.llm.http import backoff_seconds
from rag.parse.base import ParsedPaper, ParseError, Parser, ParserUnavailableError
from rag.parse.tei import tei_to_parsed

CONNECT_TIMEOUT_SECONDS = 5.0
# GROBID answers 503 when all its worker threads are busy: worth retrying.
RETRY_STATUS_CODES = {503}
REQUEST_FIELDS = [
    ("consolidateHeader", "0"),
    ("consolidateCitations", "0"),
    ("consolidateFunders", "0"),
    ("teiCoordinates", "p"),
    ("teiCoordinates", "head"),
    ("teiCoordinates", "figure"),
    ("teiCoordinates", "formula"),
]


class GrobidParser(Parser):
    name = "grobid"

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        tei_dir: Path | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.tei_dir = tei_dir  # when set, parse() keeps GROBID's raw TEI here for inspection
        self._version: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None, tei_dir: Path | None = None) -> "GrobidParser":
        settings = settings or get_settings()
        return cls(settings.grobid_url, settings.grobid_timeout_seconds, settings.grobid_max_retries, tei_dir)

    def is_alive(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=CONNECT_TIMEOUT_SECONDS)
        except requests.RequestException:
            return False
        return response.ok and response.text.strip() == "true"

    @property
    def version(self) -> str:
        if self._version is None:
            try:
                response = requests.get(f"{self.base_url}/api/version", timeout=CONNECT_TIMEOUT_SECONDS)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise ParserUnavailableError(f"GROBID not reachable at {self.base_url}: {exc}") from exc
            self._version = response.json()["version"]
        return self._version

    def process_pdf(self, pdf_path: Path) -> str:
        """Send one PDF to GROBID and return the TEI XML it produced."""
        url = f"{self.base_url}/api/processFulltextDocument"
        for attempt in range(self.max_retries + 1):
            is_last = attempt == self.max_retries
            try:
                with pdf_path.open("rb") as pdf:
                    response = requests.post(
                        url,
                        files={"input": (pdf_path.name, pdf, "application/pdf")},
                        data=REQUEST_FIELDS,
                        timeout=(CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
                    )
            except requests.ConnectionError as exc:
                if is_last:
                    raise ParserUnavailableError(f"GROBID not reachable at {self.base_url}: {exc}") from exc
                time.sleep(backoff_seconds(attempt, None))
                continue
            except requests.Timeout as exc:
                raise ParseError(f"GROBID took longer than {self.timeout_seconds}s on {pdf_path.name}") from exc

            if response.status_code in RETRY_STATUS_CODES and not is_last:
                time.sleep(backoff_seconds(attempt, response.headers.get("retry-after")))
                continue
            if response.status_code != 200 or not response.text.strip():
                raise ParseError(
                    f"GROBID returned HTTP {response.status_code} for {pdf_path.name}: {response.text[:300]}"
                )
            return response.text

        raise AssertionError("unreachable: the last attempt always returns or raises")

    def parse(self, pdf_path: Path, arxiv_id: str) -> ParsedPaper:
        tei = self.process_pdf(pdf_path)
        if self.tei_dir is not None:
            self.tei_dir.mkdir(parents=True, exist_ok=True)
            (self.tei_dir / f"{arxiv_id}.tei.xml").write_text(tei, encoding="utf-8")
        return tei_to_parsed(tei, arxiv_id, parser=f"grobid-{self.version}")
