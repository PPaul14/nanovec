"""
Three-way comparison: FlatIndex vs HNSW vs IVF+PQ.

Run from the project root:
    venv/Scripts/python.exe -m benchmarks.compare_indexes

This is the table that makes the project's argument. The three indexes are not
better or worse than each other -- they occupy different corners of the
recall / latency / memory triangle:

    Flat     exact, no build cost, O(N) per query, full memory
    HNSW     near-exact, expensive build, sublinear query, full memory + graph
    IVF+PQ   lossy, needs training, fast scan of few cells, tiny memory

Nothing here is tuned for a flattering result. At n=2000 in pure Python, flat is
expected to win on latency outright -- NumPy's vectorized C loop beats
interpreted graph traversal until N is much larger.
"""

import time

import numpy as np

from app.hnsw import HNSWIndex
from app.index import FlatIndex
from app.ivfpq import IVFPQIndex

DIM = 64
N = 2000
K = 10
N_QUERIES = 30


def build_clustered(n=N, dim=DIM, n_clusters=20, sigma=0.06, seed=11):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, (n_clusters, dim))
    per = n // n_clusters
    vectors = {}
    for ci, c in enumerate(centers):
        for j in range(per):
            vectors[f"c{ci}_v{j}"] = (c + rng.normal(0, sigma, dim)).tolist()
    return vectors


def timed(fn, repeats):
    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def main():
    vectors = build_clustered()
    ids = list(vectors.keys())
    query_ids = ids[::67][:N_QUERIES]
    print(f"dataset: {len(vectors)} vectors, dim={DIM}, {N_QUERIES} queries, k={K}\n")

    # ---------------- build ----------------
    t0 = time.perf_counter()
    flat = FlatIndex(dim=DIM)
    for vid, v in vectors.items():
        flat.insert(vid, v)
    flat_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    hnsw = HNSWIndex(dim=DIM, metric="cosine")
    for vid, v in vectors.items():
        hnsw.insert(vid, v)
    hnsw_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    ivf = IVFPQIndex(dim=DIM, nlist=32, m=8, metric="cosine", seed=0)
    ivf.train(np.array(list(vectors.values()), dtype=np.float32))
    for vid, v in vectors.items():
        ivf.add(vid, v)
    ivf_build = time.perf_counter() - t0

    # ---------------- ground truth ----------------
    truth = {
        q: {i for i, _ in flat.search(vectors[q], k=K, metric="cosine")}
        for q in query_ids
    }

    def recall_of(search_fn):
        total = 0.0
        for q in query_ids:
            got = {i for i, _ in search_fn(vectors[q])}
            total += len(truth[q] & got) / K
        return total / len(query_ids)

    rows = []

    # flat
    lat = timed(lambda: flat.search(vectors[query_ids[0]], k=K, metric="cosine"), 30)
    rows.append(
        ("Flat (exact)", 1.000, sorted(lat)[len(lat) // 2], flat_build,
         N * DIM * 4, "-")
    )

    # hnsw
    for ef in (25, 100):
        lat = timed(lambda: hnsw.search(vectors[query_ids[0]], k=K, ef=ef), 30)
        r = recall_of(lambda v, ef=ef: hnsw.search(v, k=K, ef=ef))
        edges = sum(len(x) for lay in hnsw.neighbors.values() for x in lay.values())
        # vectors + ~8 bytes per directed edge reference
        mem = N * DIM * 4 + edges * 8
        rows.append((f"HNSW (ef={ef})", r, sorted(lat)[len(lat) // 2], hnsw_build, mem, "-"))

    # ivf+pq
    for nprobe in (1, 4, 16):
        lat = timed(lambda: ivf.search(vectors[query_ids[0]], k=K, nprobe=nprobe), 30)
        r = recall_of(lambda v, np_=nprobe: ivf.search(v, k=K, nprobe=np_))
        mem = ivf.memory_bytes()["total"]
        rows.append(
            (f"IVF+PQ (nprobe={nprobe})", r, sorted(lat)[len(lat) // 2], ivf_build,
             mem, f"{ivf.compression_ratio():.0f}x")
        )

    # ---------------- report ----------------
    print(f"{'index':>24} | {'Recall@10':>9} | {'p50 ms':>7} | {'build s':>8} | "
          f"{'memory KB':>10} | {'vec compr':>9}")
    print("-" * 88)
    for name, recall, p50, build, mem, compr in rows:
        print(f"{name:>24} | {recall:>9.3f} | {p50:>7.3f} | {build:>8.2f} | "
              f"{mem/1024:>10.1f} | {compr:>9}")

    m = ivf.memory_bytes()
    print("\nIVF+PQ memory breakdown:")
    print(f"  codes            : {m['codes']/1024:8.1f} KB   ({ivf.m} bytes/vector)")
    print(f"  codebooks        : {m['codebooks']/1024:8.1f} KB   (fixed cost)")
    print(f"  coarse centroids : {m['coarse_centroids']/1024:8.1f} KB   (fixed cost)")
    print(f"  float32 would be : {m['raw_float32_equivalent']/1024:8.1f} KB")
    print(
        "\n  Note: fixed costs dominate at this N. Per-vector compression is "
        f"{ivf.compression_ratio():.0f}x,\n  but codebooks are "
        f"{m['codebooks']/max(m['total'],1)*100:.0f}% of total here -- that share "
        "shrinks toward zero as N grows.\n"
    )


if __name__ == "__main__":
    main()