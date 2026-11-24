"""Step 24: is /search still responsive while a sync job runs? Run against `rag serve`.

    uv run python experiments/stage3/sync_latency.py http://127.0.0.1:8765 COLLECTION FOLDER

Measures /search (hybrid, papers collection) for 10 s idle, then registers FOLDER in COLLECTION
(starting a sync job) and keeps searching until the job finishes. Prints p50/p95/max for both,
the job's timings, and any search that failed. Removes the folder again afterwards.
"""

import json
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx

QUERIES = [
    "why divide attention scores by sqrt d_k",
    "batch normalization internal covariate shift",
    "adam bias correction",
    "variational lower bound reparameterization",
]


def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def hammer(client, stop, out, errors):
    i = 0
    while not stop.is_set():
        start = time.perf_counter()
        r = client.post("/search", json={"query": QUERIES[i % 4], "k": 5, "collections": ["papers"]})
        (out if r.status_code == 200 else errors).append((time.perf_counter() - start) * 1000)
        i += 1


def main() -> int:
    base, collection, folder = sys.argv[1], sys.argv[2], str(Path(sys.argv[3]).resolve())
    with httpx.Client(base_url=base, timeout=60, trust_env=False) as client:
        idle, idle_errors = [], []
        stop = threading.Event()
        t = threading.Thread(target=hammer, args=(client, stop, idle, idle_errors))
        t.start()
        time.sleep(10)
        stop.set()
        t.join()

        during, during_errors = [], []
        stop = threading.Event()
        t = threading.Thread(target=hammer, args=(client, stop, during, during_errors))
        t.start()
        job = client.post(f"/collections/{collection}/folders", json={"path": folder}).json()["job"]
        while True:
            view = client.get(f"/jobs/{job['id']}").json()
            if view["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.2)
        stop.set()
        t.join()
        removed = client.delete(f"/collections/{collection}/folders", params={"path": folder}).json()
        while client.get(f"/jobs/{removed['job']['id']}").json()["status"] not in ("done", "failed", "cancelled"):
            time.sleep(0.2)

    summary = {
        "idle": {
            "n": len(idle),
            "p50_ms": round(statistics.median(idle), 1),
            "p95_ms": round(pct(idle, 95), 1),
            "max_ms": round(max(idle), 1),
            "errors": len(idle_errors),
        },
        "during_sync": {
            "n": len(during),
            "p50_ms": round(statistics.median(during), 1),
            "p95_ms": round(pct(during, 95), 1),
            "max_ms": round(max(during), 1),
            "errors": len(during_errors),
        },
        "job": {
            "status": view["status"],
            "added": len(view["result"]["added"]),
            "seconds": view["result"]["seconds"],
            "n_embedded": view["result"]["n_embedded"],
            "swap": view["result"]["swap"],
        },
        "removed_after": removed["documents_removed"],
    }
    Path(__file__).with_name("sync_latency.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
