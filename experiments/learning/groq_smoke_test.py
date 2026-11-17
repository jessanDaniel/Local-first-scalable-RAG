"""Phase 11 — isolated smoke test for GroqLLM, no retrieval/context involved.
Just: does the interface + API call actually work end-to-end?"""

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

from rag.llm.groq import GroqLLM
from rag.llm.ollama import print_generation_debug


def main() -> None:
    llm = GroqLLM()
    result = llm.generate("In one sentence, what is retrieval-augmented generation?")
    print_generation_debug(result)


if __name__ == "__main__":
    main()