"""Step 19: latency of a running `rag serve`, over real HTTP.

    rag serve --port 8765 &
    uv run python experiments/stage3/api_latency.py http://127.0.0.1:8765

/search: every papers question, warm, rerank off and fused -> p50 / p95 per request.
/ask: 6 papers questions from position N (2nd argument, default 10; not ones api_parity.py just asked,
because Ollama reuses a cached prompt), streamed -> time to the sources event, first token, done.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

from rag.eval.sets import load_set, set_path

OUT = Path(__file__).with_name("api_latency.json")


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    questions = load_set(set_path("papers"))
    results: dict = {"base": base}
    with httpx.Client(base_url=base, timeout=120, trust_env=False) as client:
        results["health"] = client.get("/health").json()["service"]
        for rerank in ("off", "fused"):
            times, server_ms = [], []
            for q in questions:
                start = time.perf_counter()
                body = client.post(
                    "/search", json={"query": q.question, "k": 10, "collections": ["papers"], "rerank": rerank}
                ).json()
                times.append((time.perf_counter() - start) * 1000)
                server_ms.append(body["timings_ms"]["total_ms"])
            results[f"search_{rerank}"] = {
                "n": len(times),
                "p50_ms": round(statistics.median(times), 1),
                "p95_ms": round(pct(times, 95), 1),
                "server_p50_ms": round(statistics.median(server_ms), 1),
            }
            print(f"/search rerank={rerank}: {results[f'search_{rerank}']}")

        asks = []
        for q in [q for q in questions if q.answerable][int(sys.argv[2]) if len(sys.argv) > 2 else 10 :][
            :6
        ]:  # 10+: not asked in api_parity.py (Ollama caches prompts)
            start = time.perf_counter()
            marks: dict[str, float] = {}
            event = None
            with client.stream("POST", "/ask", json={"question": q.question, "collections": ["papers"]}) as r:
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                        if event not in marks:
                            marks[event] = time.perf_counter() - start
                    elif line.startswith("data: ") and event == "done":
                        done = json.loads(line[6:])
            row = {
                "id": q.id,
                "sources_s": round(marks["sources"], 2),
                "first_token_s": round(marks["token"], 2),
                "done_s": round(marks["done"], 2),
                "answer_tokens": done["generation"]["completion_tokens"],
                # Ollama's own report: tokens in the prompt, seconds reading it, seconds loading the model
                "prompt_tokens": done["generation"]["prompt_tokens"],
                "ollama_prompt_s": done["generation"]["prompt_seconds"],
                "ollama_load_s": done["generation"]["load_seconds"],
            }
            asks.append(row)
            print(f"/ask {row}")
        results["ask"] = asks
        for key in ("sources_s", "first_token_s", "done_s"):
            results[f"ask_median_{key}"] = round(statistics.median(r[key] for r in asks), 2)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k.startswith("ask_median")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
