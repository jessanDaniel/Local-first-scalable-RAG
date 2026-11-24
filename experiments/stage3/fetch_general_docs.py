"""Step 21b: fetch the general-documents test set (9 US federal publications, public domain under
17 U.S.C. 105; approved by the user 2026-11-24) into data/raw/incoming/general/ and write a manifest
(file, bytes, sha256, URL) next to this script. Files that are already there are not fetched again.

    uv run python experiments/stage3/fetch_general_docs.py
"""

import hashlib
import sys
from pathlib import Path

import httpx

from rag.config import get_settings

DOCS = [
    ("tax/irs-p463-travel-gift-car-expenses-2025.pdf", "https://www.irs.gov/pub/irs-prior/p463--2025.pdf"),
    (
        "hr/opm-handbook-childbirth-adoption-foster-care-2025.pdf",
        "https://www.opm.gov/chcoc/transmittals/2025/Rev%20CPM%202025-04%20Handbook%20on%20Flexibilities%20for"
        "%20Childbirth%20Adoption%20and%20Foster%20Care.pdf",
    ),
    (
        "hr/opm-handbook-elder-care-2015.pdf",
        "https://www.opm.gov/policy-data-oversight/pay-leave/leave-administration/fact-sheets/"
        "handbook-on-workplace-flexibilities-and-work-life-programs-for-elder-care.pdf",
    ),
    (
        "safety/osha-3151-personal-protective-equipment.pdf",
        "https://www.osha.gov/sites/default/files/publications/osha3151.pdf",
    ),
    (
        "tech/nist-sp-800-63b-4-digital-identity.pdf",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf",
    ),
    (
        "safety/fema-are-you-ready-guide-2020.pdf",
        "https://www.ready.gov/sites/default/files/2021-03/are-you-ready-guide.pdf",
    ),
    (
        "tech/nasa-ai-for-lessons-learned-2024.pdf",
        "https://ntrs.nasa.gov/api/citations/20240006820/downloads/Ideas_to_%20Apply_AI_to_NASA_Lessons_Learned.pdf",
    ),
    (
        "tech/autopen-template-decision-document.docx",
        "https://www.idmanagement.gov/docs/autopen-template-decision-document.docx",
    ),
    (
        "hr/opm-time-off-for-safe-leave-purposes.html",
        "https://www.opm.gov/policy-data-oversight/pay-leave/leave-administration/fact-sheets/"
        "time-off-for-safe-leave-purposes/",
    ),
]
# Generic on purpose: nothing about the user goes into request headers.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rag-research/0.1)"}


def main() -> int:
    root = get_settings().data_dir / "raw" / "incoming" / "general"
    lines = []
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120, trust_env=False) as client:
        for name, url in DOCS:
            path = root / name
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            data = path.read_bytes()
            lines.append(f"{name}|{len(data)}|{hashlib.sha256(data).hexdigest()}|{url}")
            print(f"{len(data):>9}  {name}")
    Path(__file__).with_name("general_docs_manifest.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
