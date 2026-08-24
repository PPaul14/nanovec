"""
Diagnostic benchmark: Recall@10 vs ef, on uniform vs clustered data.

Run from the project root (vector-db/):
    venv/Scripts/python.exe -m benchmarks.ef_curve

Purpose:
  - Uniform random data is artificially easy in high dimensions (points are
    near-equidistant), so recall saturates at 1.0 and tells you nothing.
  - Clustered data resembles real embeddings and is where approximation error
    actually shows up.
  - If recall climbs with ef  -> tuning issue (graph is fine, search too shallow).
  - If recall stays flat      -> structural issue in graph construction.
"""

import time
import numpy as np

from app.index import FlatIndex
from app.hnsw import HNSWIndex

K = 10
EF_VALUES = [10, 25, 50, 100, 200, 400]


def build_uniform(dim=32, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return {f"v{i}": rng.uniform(-1, 1, dim).tolist() for i in range(n)}


def build_clustered(dim=32, n_clusters=10, per_cluster=100, sigma=0.05, seed=7):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, (n_clusters, dim))
    vectors = {}
    for ci, center in enumerate(centers):
        for j in range(per_cluster):
            vectors[f"c{ci}_v{j}"] = (center + rng.normal(0, sigma, dim)).tolist()
    return vectors


def run(name, vectors, dim):
    flat = FlatIndex(dim=dim)
    hnsw = HNSWIndex(dim=dim, metric="cosine")
    for vid, vec in vectors.items():
        flat.insert(vid, vec)
        hnsw.insert(vid, vec)

    query_ids = list(vectors.keys())[::37][:30]
    truth = {
        q: {i for i, _ in flat.search(vectors[q], k=K, metric="cosine")}
        for q in query_ids
    }

    print(f"\n=== {name} (n={len(vectors)}, dim={dim}) ===")
    print(f"{'ef':>6} | {'Recall@10':>10} | {'p50 (ms)':>9} | {'QPS':>8}")
    print("-" * 44)

    for ef in EF_VALUES:
        recalls, lat = [], []
        for q in query_ids:
            t0 = time.perf_counter()
            approx = {i for i, _ in hnsw.search(vectors[q], k=K, ef=ef)}
            lat.append((time.perf_counter() - t0) * 1000)
            recalls.append(len(truth[q] & approx) / K)
        avg = sum(recalls) / len(recalls)
        p50 = sorted(lat)[len(lat) // 2]
        qps = 1000 / (sum(lat) / len(lat))
        print(f"{ef:>6} | {avg:>10.3f} | {p50:>9.3f} | {qps:>8.1f}")

    lat = []
    for q in query_ids:
        t0 = time.perf_counter()
        flat.search(vectors[q], k=K, metric="cosine")
        lat.append((time.perf_counter() - t0) * 1000)
    p50 = sorted(lat)[len(lat) // 2]
    print("-" * 44)
    print(f"{'flat':>6} | {1.000:>10.3f} | {p50:>9.3f} | {1000/(sum(lat)/len(lat)):>8.1f}")


if __name__ == "__main__":
    print("HNSW diagnostic: recall vs ef")
    run("Uniform random", build_uniform(), dim=32)
    run("Clustered (sigma=0.05)", build_clustered(), dim=32)
    print("\nDone. If clustered recall rises with ef -> tuning. If flat -> graph bug.\n")