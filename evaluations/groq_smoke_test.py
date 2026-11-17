"""Phase 11 — isolated smoke test for GroqLLM, no retrieval/context involved.
Just: does the interface + API call actually work end-to-end?"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.llm.groq import GroqLLM
from src.llm.ollama import print_generation_debug


def main() -> None:
    llm = GroqLLM()
    result = llm.generate("In one sentence, what is retrieval-augmented generation?")
    print_generation_debug(result)


if __name__ == "__main__":
    main()