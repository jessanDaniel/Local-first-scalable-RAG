"""Step 25: which dense index, at which size? Recall against exact search, memory, latency.

    uv run python experiments/stage4/index_bench.py real            # the real corpus (1,444 chunks)
    uv run python experiments/stage4/index_bench.py calibrate       # synthetic generator vs real vectors
    uv run python experiments/stage4/index_bench.py all 100000 1000000

`all` prepares each size (synthetic base vectors, queries, exact top-100 with and without a
collection filter) and then runs every index family in its own process, at low priority, appending
rows to index_bench.csv. One process per family keeps the memory numbers honest, and chromadb and
torch can't share a process (two OpenMP runtimes).

Synthetic vectors: real embeddings sit in a narrow, anisotropic cone with document-sized clusters;
uniform random vectors have neither and make ANN look worse than it is. So: fit a Gaussian (mean,
full covariance) to the real chunk vectors, draw one centre per "document" of 50 chunks from it,
draw each chunk around its centre with the same covariance scaled by `spread`, normalise. `spread`
is calibrated so that, at the real corpus size, the nearest-neighbour similarities match the real
ones (`calibrate`). Queries are drawn the same way, in random existing documents.

A collection filter keeps whole documents: 1% and 10% of them, the way a collection holds whole
files. Filtered recall is measured against exact search over the same subset.

Large files go to data/bench/ (gitignored; delete after). Nothing here changes the app.
"""

import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "bench"
OUT_CSV = Path(__file__).with_name("index_bench.csv")
REAL_CSV = Path(__file__).with_name("index_bench_real.csv")
CALIBRATION = Path(__file__).with_name("index_bench_calibration.json")
K = 100  # the app asks the dense index for its top 100 (hybrid_candidates)
PER_DOC = 50  # chunks per synthetic document (the real corpus: 1,444 chunks in 27 documents)
N_QUERIES = 1000
N_FILTER_QUERIES = 300
WARMUP = 50
SEED = 25
FIELDS = [
    "corpus", "n", "index", "params", "filter", "recall@10", "recall@100", "p50_ms", "p95_ms", "p99_ms",
    "build_s", "index_mb", "peak_rss_mb", "threads", "load1", "when",
]  # fmt: skip


# --- data ----------------------------------------------------------------------------------------


def real_vectors() -> tuple[np.ndarray, list[str]]:
    """The current dense build's vectors (exact index: reconstructable) and chunk ids."""
    import faiss

    from rag.config import get_settings
    from rag.index.current import current_build_dir

    build = current_build_dir(get_settings().data_dir / "indexes" / "dense")
    index = faiss.read_index(str(build / "index.faiss"))
    vectors = index.reconstruct_n(0, index.ntotal)
    return np.ascontiguousarray(vectors, dtype=np.float32), json.loads((build / "chunk_ids.json").read_text())


class Generator:
    def __init__(self, real: np.ndarray, spread: float, seed: int = SEED):
        self.mean = real.mean(axis=0)
        cov = np.cov(real, rowvar=False) + 1e-6 * np.eye(real.shape[1])
        self.chol = np.linalg.cholesky(cov).astype(np.float32)
        self.spread = spread
        self.rng = np.random.default_rng(seed)
        self.dim = real.shape[1]

    def gaussian(self, n: int) -> np.ndarray:
        return self.rng.standard_normal((n, self.dim), dtype=np.float32) @ self.chol.T

    def centres(self, n_docs: int) -> np.ndarray:
        return self.mean + self.gaussian(n_docs)

    def around(self, centres: np.ndarray, doc_of: np.ndarray) -> np.ndarray:
        out = np.empty((len(doc_of), self.dim), dtype=np.float32)
        for start in range(0, len(doc_of), 100_000):  # float32, in blocks: 1M x 384 at once is 1.5 GB
            rows = doc_of[start : start + 100_000]
            block = centres[rows] + self.spread * self.gaussian(len(rows))
            out[start : start + len(rows)] = block / np.linalg.norm(block, axis=1, keepdims=True)
        return out


def neighbour_stats(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Median cosine to the nearest and 10th-nearest other vector, and between random pairs."""
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    top = -np.sort(-sims, axis=1)
    i, j = rng.integers(0, len(x), 20_000), rng.integers(0, len(x), 20_000)
    keep = i != j
    return {
        "top1": round(float(np.median(top[:, 0])), 4),
        "top10": round(float(np.median(top[:, 9])), 4),
        "random_pair": round(float(np.median(np.sum(x[i[keep]] * x[j[keep]], axis=1))), 4),
    }


def calibrate() -> dict:
    real, _ = real_vectors()
    rng = np.random.default_rng(SEED)
    target = neighbour_stats(real, rng)
    n_docs = math.ceil(len(real) / PER_DOC)
    doc_of = np.arange(len(real)) // PER_DOC
    tried = []
    for spread in np.round(np.arange(0.05, 1.01, 0.05), 2):
        gen = Generator(real, float(spread))
        synth = gen.around(gen.centres(n_docs), doc_of)
        stats = neighbour_stats(synth, np.random.default_rng(SEED))
        tried.append({"spread": float(spread), **stats})
    best = min(tried, key=lambda t: abs(t["top1"] - target["top1"]) + abs(t["top10"] - target["top10"]))
    result = {"real": target, "chosen": best, "tried": tried, "n_real": len(real)}
    CALIBRATION.write_text(json.dumps(result, indent=1))
    print(f"real      top1 {target['top1']}  top10 {target['top10']}  random pair {target['random_pair']}")
    print(
        f"synthetic top1 {best['top1']}  top10 {best['top10']}  random pair {best['random_pair']}  "
        f"(spread {best['spread']})"
    )
    return result


def prepare(n: int) -> Path:
    """Base vectors, queries, filter subsets and exact top-K for one size (skipped if present)."""
    import faiss

    folder = BENCH / str(n)
    if (folder / "gt.npy").exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    real, _ = real_vectors()
    spread = json.loads(CALIBRATION.read_text())["chosen"]["spread"]
    gen = Generator(real, spread)
    n_docs = math.ceil(n / PER_DOC)
    centres = gen.centres(n_docs)
    doc_of = np.arange(n) // PER_DOC
    base = gen.around(centres, doc_of)
    queries = gen.around(centres, gen.rng.integers(0, n_docs, N_QUERIES))
    np.save(folder / "base.npy", base)
    np.save(folder / "queries.npy", queries)
    exact = faiss.IndexFlatIP(base.shape[1])
    exact.add(base)
    _, gt = exact.search(queries, K)
    np.save(folder / "gt.npy", gt)
    for pct in (1, 10):
        docs = gen.rng.choice(n_docs, max(1, n_docs * pct // 100), replace=False)
        rows = np.flatnonzero(np.isin(doc_of, docs)).astype(np.int64)
        sel = faiss.IDSelectorBatch(rows)
        _, gt_f = exact.search(queries[:N_FILTER_QUERIES], K, params=faiss.SearchParameters(sel=sel))
        np.save(folder / f"subset{pct}.npy", rows)
        np.save(folder / f"gt_f{pct}.npy", gt_f)
    print(f"prepared {n:,} vectors in {folder}")
    return folder


# --- measuring -----------------------------------------------------------------------------------


def recall(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth, strict=True)]))


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS


def index_mb(index, folder: Path) -> float:
    import faiss

    path = folder / "size.index"
    faiss.write_index(index, str(path))
    size = path.stat().st_size / 1e6
    path.unlink()
    return round(size, 1)


def time_queries(search, queries: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """One query at a time, the way the app searches. Returns the found ids and per-query ms."""
    for q in queries[:WARMUP]:
        search(q.reshape(1, -1))
    found, times = [], []
    for q in queries:
        start = time.perf_counter()
        ids = search(q.reshape(1, -1))
        times.append((time.perf_counter() - start) * 1000)
        found.append(ids)
    return np.array(found), times


def row(corpus, n, name, params, flt, found, truth, times, build_s, size_mb) -> dict:
    faiss = sys.modules.get("faiss")  # not imported by the MPS and Chroma jobs
    return {
        "corpus": corpus,
        "n": n,
        "index": name,
        "params": params,
        "filter": flt,
        "recall@10": round(recall(found, truth, 10), 4),
        "recall@100": round(recall(found, truth, K), 4),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "build_s": round(build_s, 1),
        "index_mb": size_mb,
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "threads": faiss.omp_get_max_threads() if faiss else os.cpu_count(),
        "load1": round(os.getloadavg()[0], 1),
        "when": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def append(rows: list[dict], path: Path = OUT_CSV) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(
            f"{r['n']:>9,} {r['index']:<16} {r['params']:<26} {r['filter']:<5} "
            f"r@10 {r['recall@10']:.3f} r@100 {r['recall@100']:.3f}  p50 {r['p50_ms']:7.2f} p95 {r['p95_ms']:7.2f} ms  "
            f"{r['index_mb']:>8} MB  build {r['build_s']} s",
            flush=True,
        )


# --- index families (FAISS) ----------------------------------------------------------------------


def families(d: int, n: int) -> dict:
    """name -> (builder, [(params label, search params factory or None)], filtered search config)."""
    import faiss

    ip = faiss.METRIC_INNER_PRODUCT
    nlist = 4096 if n >= 500_000 else 1024 if n >= 50_000 else 32

    def hnsw(m):
        index = faiss.IndexHNSWFlat(d, m, ip)
        index.hnsw.efConstruction = 200
        return index

    def hnsw_sq8():
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 32, ip)
        index.hnsw.efConstruction = 200
        return index

    def ivfpq():
        return faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 48, 8, ip)

    def refine_flat():
        return faiss.IndexRefineFlat(ivfpq())

    def refine_sq8():
        return faiss.IndexRefine(ivfpq(), faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip))

    def hnsw_params(ef, sel=None):
        return faiss.SearchParametersHNSW(efSearch=ef, sel=sel)

    hnsw_sweep = [(f"efSearch={ef}", lambda ef=ef, sel=None: hnsw_params(ef, sel)) for ef in (16, 32, 64, 128, 256)]
    nprobes = (4, 8, 16, 32) if nlist <= 32 else (8, 16, 32, 64)
    ivf_sweep = [(f"nprobe={p}", lambda p=p, sel=None: faiss.SearchParametersIVF(nprobe=p, sel=sel)) for p in nprobes]

    def refined(p, sel=None, k_factor=4):
        base = faiss.SearchParametersIVF(nprobe=p, sel=sel)
        return faiss.IndexRefineSearchParameters(k_factor=k_factor, base_index_params=base)

    refine_sweep = [(f"nprobe={p},k_factor=4", lambda p=p, sel=None: refined(p, sel)) for p in nprobes]
    plain = [("", lambda sel=None: faiss.SearchParameters(sel=sel) if sel is not None else None)]
    mid_ivf = nprobes[2]
    refine_sq8_filtered = (f"nprobe={mid_ivf},k_factor=4", lambda sel=None: refined(mid_ivf, sel))
    return {
        "flat": (lambda: faiss.IndexFlatIP(d), plain, plain[0]),
        "sq8": (lambda: faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip), plain, plain[0]),
        "hnsw_m16": (lambda: hnsw(16), hnsw_sweep, hnsw_sweep[3]),
        "hnsw_m32": (lambda: hnsw(32), hnsw_sweep, hnsw_sweep[3]),
        "hnsw_sq8": (hnsw_sq8, hnsw_sweep, hnsw_sweep[3]),
        "ivfpq": (ivfpq, ivf_sweep, ivf_sweep[2]),
        "ivfpq_refine_flat": (refine_flat, refine_sweep, refine_sweep[2]),
        "ivfpq_refine_sq8": (refine_sq8, refine_sweep, refine_sq8_filtered),
    }  # fmt: skip


def build(builder, base: np.ndarray, rng: np.random.Generator):
    index = builder()
    start = time.perf_counter()
    if not index.is_trained:
        sample = base[rng.choice(len(base), min(len(base), 200_000), replace=False)]
        index.train(sample)
    index.add(base)
    return index, time.perf_counter() - start


def run_family(n: int, family: str) -> None:
    import faiss

    folder = BENCH / str(n)
    base, queries, gt = np.load(folder / "base.npy"), np.load(folder / "queries.npy"), np.load(folder / "gt.npy")
    builder, sweep, filtered = families(base.shape[1], n)[family]
    index, build_s = build(builder, base, np.random.default_rng(SEED))
    size = index_mb(index, folder)
    rows = []
    for label, make in sweep:
        params = make()
        found, times = time_queries(lambda q, p=params: index.search(q, K, params=p)[1][0], queries)
        rows.append(row("synthetic", n, family, label, "none", found, gt, times, build_s, size))
    label, make = filtered
    for pct in (1, 10):
        sel = faiss.IDSelectorBatch(np.load(folder / f"subset{pct}.npy"))
        params = make(sel=sel)
        qs, truth = queries[:N_FILTER_QUERIES], np.load(folder / f"gt_f{pct}.npy")
        found, times = time_queries(lambda q, p=params: index.search(q, K, params=p)[1][0], qs)
        rows.append(row("synthetic", n, family, label, f"{pct}%", found, truth, times, build_s, size))
    append(rows)


def run_mps(n: int) -> None:
    """Exact search as one matrix product on the Apple GPU (float32, all vectors resident)."""
    import torch  # no faiss in this process: faiss + torch's OpenMP runtimes abort together here

    folder = BENCH / str(n)
    base, queries, gt = np.load(folder / "base.npy"), np.load(folder / "queries.npy"), np.load(folder / "gt.npy")
    start = time.perf_counter()
    matrix = torch.from_numpy(base).to("mps")
    torch.mps.synchronize()
    build_s = time.perf_counter() - start

    def search(q):
        scores = matrix @ torch.from_numpy(q[0]).to("mps")
        return torch.topk(scores, K).indices.cpu().numpy()

    found, times = time_queries(search, queries)
    size = round(base.nbytes / 1e6, 1)
    append([row("synthetic", n, "mps_exact", "float32", "none", found, gt, times, build_s, size)])


def run_chroma(n: int) -> None:
    """Chroma (embedded vector database, HNSW inside), default-like HNSW settings, telemetry off."""
    import shutil

    import chromadb
    from chromadb.config import Settings

    folder = BENCH / str(n)
    base, queries, gt = np.load(folder / "base.npy"), np.load(folder / "queries.npy"), np.load(folder / "gt.npy")
    path = folder / "chroma"
    shutil.rmtree(path, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    config = {"hnsw": {"space": "ip", "ef_construction": 100, "ef_search": 100, "max_neighbors": 16}}
    collection = client.create_collection("bench", configuration=config, embedding_function=None)
    batch = client.get_max_batch_size()
    start = time.perf_counter()
    for i in range(0, n, batch):
        collection.add(ids=[str(j) for j in range(i, min(n, i + batch))], embeddings=base[i : i + batch])
    build_s = time.perf_counter() - start

    def search(q):
        result = collection.query(query_embeddings=q, n_results=K, include=[])
        return np.array([int(x) for x in result["ids"][0]])

    found, times = time_queries(search, queries)
    disk_mb = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)
    append([row("synthetic", n, "chroma", "M=16,efC=100,ef=100 (disk MB)", "none", found, gt, times, build_s, disk_mb)])
    del client
    shutil.rmtree(path, ignore_errors=True)


# --- the real corpus -----------------------------------------------------------------------------


def run_real() -> None:
    """Every family on the real chunk vectors: recall vs exact on the eval questions, and the
    eval sets' dense and hybrid Hit@k / MRR with that index in place of the exact one."""
    import faiss

    from rag.eval.gate import retrieval_snapshot
    from rag.eval.sets import load_set, set_path
    from rag.index.dense import DenseIndex
    from rag.pipeline import Pipeline

    real, chunk_ids = real_vectors()
    pipeline = Pipeline()
    dense = pipeline.dense()  # the hybrid searcher below shares this object: swapping .index swaps both
    manifest = dense.index.manifest

    texts = [q.question for name in ("papers", "hr-demo", "injection", "gov-docs") for q in load_set(set_path(name))]
    queries = np.vstack([dense.embedder.embed_query(t) for t in texts]).astype(np.float32)
    exact = faiss.IndexFlatIP(real.shape[1])
    exact.add(real)
    _, gt = exact.search(queries, K)

    class Index(DenseIndex):
        """DenseIndex whose filtered search passes the family's own search parameters (the app's
        generic SearchParameters is refused by IVF indexes)."""

        def __init__(self, index, make_params):
            super().__init__(index, chunk_ids, manifest)
            self.make_params = make_params

        def search(self, query_vector, k, allowed_chunk_ids=None):
            query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
            sel = None
            limit = min(k, self.index.ntotal)
            if allowed_chunk_ids is not None:
                rows = np.array([self.row_of[c] for c in allowed_chunk_ids if c in self.row_of], dtype=np.int64)
                sel, limit = faiss.IDSelectorBatch(rows), min(k, len(rows))
            scores, found = self.index.search(query, limit, params=self.make_params(sel=sel))
            return [(self.chunk_ids[r], float(v)) for r, v in zip(found[0], scores[0], strict=True) if r >= 0]

    out = []
    for family, (builder, _sweep, chosen) in families(real.shape[1], len(real)).items():
        index, build_s = build(builder, real, np.random.default_rng(SEED))
        label, make = chosen
        params = make()
        found, times = time_queries(lambda q, i=index, p=params: i.search(q, K, params=p)[1][0], queries)
        r = row("real", len(real), family, label, "none", found, gt, times, build_s, index_mb(index, BENCH))
        dense.index = Index(index, make)
        for mode in ("dense", "hybrid"):
            snapshot = retrieval_snapshot(pipeline.conn, pipeline.searcher(mode, "off"))
            for name, snap in snapshot.items():
                m = snap["metrics"]
                r[f"{mode}:{name}"] = f"{m['hit@1']}/{m['hit@5']}/{m['mrr']:.3f}"
        out.append(r)
        print(family, label, f"recall@100 {r['recall@100']:.3f}", {k: v for k, v in r.items() if ":" in k}, flush=True)
    fields = FIELDS + [k for k in out[0] if ":" in k]
    with REAL_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    pipeline.close()


# --- driver --------------------------------------------------------------------------------------


def run_all(sizes: list[int]) -> None:
    me = [sys.executable, __file__]
    if not CALIBRATION.exists():
        calibrate()
    for n in sizes:
        prepare(n)
        jobs = [["run", str(n), f] for f in families(384, n)] + [["mps", str(n)], ["chroma", str(n)]]
        for job in jobs:
            start = time.perf_counter()
            done = subprocess.run(["nice", "-n", "10", *me, *job], cwd=ROOT)
            print(f"-- {' '.join(job)}: exit {done.returncode}, {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["help"]
    if command == "real":
        BENCH.mkdir(parents=True, exist_ok=True)
        run_real()
    elif command == "calibrate":
        calibrate()
    elif command == "prepare":
        prepare(int(rest[0]))
    elif command == "run":
        run_family(int(rest[0]), rest[1])
    elif command == "mps":
        run_mps(int(rest[0]))
    elif command == "chroma":
        run_chroma(int(rest[0]))
    elif command == "all":
        run_all([int(x) for x in rest])
    else:
        print(__doc__)
