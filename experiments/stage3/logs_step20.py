"""Step 20 proof: real requests against a running `rag serve`, then explained from the log alone.

    rag serve --port 8765 &
    uv run python experiments/stage3/logs_step20.py http://127.0.0.1:8765

1. Sends a handful of chosen requests (a normal answer, a known wrong retrieval, a refusal, an HR
   question, a search), each with a fixed X-Request-ID, so `rag logs show <id>` can be read next.
2. Checks every record: the named stages + other_ms == total_ms, the request id matches.
3. Overhead: warm /search p50 over the 45 papers questions (step 19 measured 9 ms without logs), and
   the cost of RequestLog.write itself on the real records just written.
"""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx

from rag.config import get_settings
from rag.eval.sets import load_set, set_path
from rag.requestlog import RequestLog, read_records

OUT = Path(__file__).with_name("logs_step20.json")

ASKS = [
    # (request id, question id or literal question, collection, rerank)
    ("p20-normal", "attn_sqrt_dk", "papers", None),
    ("p20-vae-replace", "vae_reparam", "papers", "replace"),
    ("p20-vae-fused", "vae_reparam", "papers", None),
    ("p20-refusal", "What GPU did the Adam authors train on?", "papers", None),
    ("p20-hr", "How many vacation days do employees get in their first year?", "hr-demo", None),
]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    questions = {q.id: q for q in load_set(set_path("papers"))}
    results: dict = {"base": base, "asks": []}
    with httpx.Client(base_url=base, timeout=180, trust_env=False) as client:
        for request_id, question, collection, rerank in ASKS:
            text = questions[question].question if question in questions else question
            body = {"question": text, "collections": [collection], "stream": True}
            if rerank:
                body["rerank"] = rerank
            start = time.perf_counter()
            with client.stream("POST", "/ask", json=body, headers={"X-Request-ID": request_id}) as r:
                last_event = None
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        last_event = line[7:]
            results["asks"].append(
                {"request_id": request_id, "last_event": last_event, "client_s": round(time.perf_counter() - start, 2)}
            )
            print(results["asks"][-1])
        client.post(
            "/search",
            json={"query": questions["vae_reparam"].question, "k": 5, "collections": ["papers"], "rerank": "replace"},
            headers={"X-Request-ID": "p20-search-vae"},
        )
        client_ms = []
        for q in questions.values():
            start = time.perf_counter()
            client.post("/search", json={"query": q.question, "k": 10, "collections": ["papers"]})
            client_ms.append((time.perf_counter() - start) * 1000)
        results["search_warm_p50_ms"] = round(statistics.median(client_ms), 1)
        results["health_requests"] = client.get("/health").json()["requests"]

    settings = get_settings()
    records = read_records(settings.data_dir / "logs" / "requests.jsonl", settings.log_backups)
    ours = {r["request_id"]: r for r in records if r["request_id"].startswith("p20-")}
    checks = []
    for r in ours.values():
        t = r["timings_ms"]
        top = t["total_ms"] - t["other_ms"]
        checks.append({"request_id": r["request_id"], "status": r["status"], "top_level_ms": top, **t})
    results["records"] = checks

    # The write itself, on real records (fresh file, same size of records).
    with tempfile.TemporaryDirectory() as tmp:
        log = RequestLog(Path(tmp) / "requests.jsonl")
        sample = records[-50:]
        start = time.perf_counter()
        for record in sample:
            log.write(record)
        results["write_ms_per_record"] = round((time.perf_counter() - start) * 1000 / len(sample), 3)
        results["bytes_per_record"] = {
            "search": round(
                statistics.median(len(json.dumps(r, ensure_ascii=False)) for r in sample if r["route"] == "search")
            ),
            "ask": round(statistics.median(len(json.dumps(r, ensure_ascii=False)) for r in ours.values())),
        }
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(
        json.dumps({k: results[k] for k in ("search_warm_p50_ms", "write_ms_per_record", "bytes_per_record")}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
