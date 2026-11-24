"""Step 19 proof: does the API give the same results as the CLI, on the real data and models?

    uv run python experiments/stage3/api_parity.py            # /search parity only (no LLM)
    uv run python experiments/stage3/api_parity.py --ask      # + /ask on 10 questions (Ollama running)

/search: every question of the three gate sets, through POST /search (in-process, real pipeline)
and through the CLI's searcher (a fresh Pipeline, exactly what `rag search` builds). Same chunk ids
in the same order, same scores. Hybrid without reranking on all of them; fused reranking on the
first 20 papers questions (the reranker costs ~1.8 s each).

/ask: 10 questions through the streamed POST /ask and through the CLI's Answerer.ask. The prompt
sent to the model must be byte-identical. The answer text is compared too, but Ollama at
temperature 0 is not exactly reproducible (step 8), so a different wording is reported, not failed.
"""

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.config import get_settings
from rag.eval.sets import load_set, set_path
from rag.llm.factory import create_llm
from rag.llm.interface import LLM
from rag.pipeline import Pipeline

OUT = Path(__file__).with_name("api_parity.json")
K = 10


class Recording(LLM):
    """Passes calls to the real LLM and keeps every (system, prompt) it was sent."""

    def __init__(self, inner: LLM):
        self.inner = inner
        self.provider, self.model = inner.provider, inner.model
        self.context_window, self.max_tokens = inner.context_window, inner.max_tokens
        self.prompts: list[tuple[str | None, str]] = []

    def generate(self, prompt, system=None):
        self.prompts.append((system, prompt))
        return self.inner.generate(prompt, system)

    def generate_stream(self, prompt, system, on_token):
        self.prompts.append((system, prompt))
        return self.inner.generate_stream(prompt, system, on_token)


def sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def search_parity(client, cli: Pipeline, questions, rerank: str) -> dict:
    searcher = cli.searcher("hybrid", rerank)
    same, different = 0, []
    for q in questions:
        api = client.post(
            "/search", json={"query": q.question, "k": K, "collections": [q.collection], "rerank": rerank}
        ).json()["hits"]
        direct = searcher.search(q.question, K, [q.collection])
        api_ids = [(h["chunk_id"], round(h["score"], 9)) for h in api]
        cli_ids = [(h.chunk.chunk_id, round(h.score, 9)) for h in direct]
        if api_ids == cli_ids:
            same += 1
        else:
            different.append({"id": q.id, "api": api_ids, "cli": cli_ids})
    return {"rerank": rerank, "questions": len(questions), "identical": same, "different": different}


def ask_parity(settings, questions) -> list[dict]:
    api_llm = Recording(create_llm(settings))
    app = create_app(settings, pipeline_factory=lambda: Pipeline(settings, llm=api_llm))
    cli = Pipeline(settings, llm=Recording(create_llm(settings)))
    answerer = cli.answerer(settings.retrieval_mode, settings.ask_rerank)
    rows = []
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for q in questions:
            body = {"question": q.question, "collections": [q.collection]}
            api = sse_events(client.post("/ask", json=body).text)[-1][1]
            answer = answerer.ask(q.question, [q.collection])
            rows.append(
                {
                    "id": q.id,
                    "prompt_identical": api_llm.prompts[-1] == cli.llm.prompts[-1],
                    "system_identical": api_llm.prompts[-1][0] == cli.llm.prompts[-1][0],
                    "sources_identical": [s["chunk_ids"] for s in api["sources"]]
                    == [s.chunk_ids for s in answer.sources],
                    "answer_identical": api["text"] == answer.text,
                    "flags_api": [f["kind"] for f in api["flags"]],
                    "flags_cli": [f.kind for f in answer.flags],
                    "api_text": api["text"],
                    "cli_text": answer.text,
                }
            )
            print(f"  {q.id:28} prompt same={rows[-1]['prompt_identical']} answer same={rows[-1]['answer_identical']}")
    return rows


def main() -> int:
    settings = get_settings()
    sets = {name: [q for q in load_set(set_path(name))] for name in ("papers", "hr-demo", "injection")}
    everything = [q for qs in sets.values() for q in qs]
    results: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "k": K}

    start = time.perf_counter()
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        results["server_load_seconds"] = client.get("/health").json()["service"]["load_seconds"]
        cli = Pipeline(settings)
        results["search"] = [
            search_parity(client, cli, everything, "off"),
            search_parity(client, cli, sets["papers"][:20], "fused"),
        ]
        cli.close()
    for r in results["search"]:
        print(f"/search rerank={r['rerank']}: {r['identical']}/{r['questions']} identical")
    print(f"({time.perf_counter() - start:.0f} s)")

    if "--ask" in sys.argv:
        picked = sets["papers"][:7] + sets["hr-demo"][:3]
        results["ask"] = ask_parity(settings, picked)
        rows = results["ask"]
        for key in ("prompt_identical", "sources_identical", "answer_identical"):
            print(f"/ask {key}: {sum(r[key] for r in rows)}/{len(rows)}")

    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"saved {OUT}")
    ok = all(not r["different"] for r in results["search"]) and all(
        r["prompt_identical"] for r in results.get("ask", [])
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
