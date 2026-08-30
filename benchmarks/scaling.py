"""
Scaling benchmark: how Flat, HNSW and IVF+PQ behave as N grows.

Run from the project root:
    venv/Scripts/python.exe -m benchmarks.scaling
    venv/Scripts/python.exe -m benchmarks.scaling --sizes 1000,10000 --tag baseline

Three questions the existing n=1000 benchmarks cannot answer
------------------------------------------------------------
1. Where is the flat/HNSW crossover?
   Brute force wins at small N because a flat search is a single vectorized
   NumPy matrix multiply -- one C loop over the whole dataset. HNSW pays Python
   interpreter overhead for every node it visits, which is a large constant
   factor. Its O(log N) advantage has to grow past that constant before it wins.
   This locates the N where that actually happens instead of assuming it.

2. Does a real recall/latency curve appear?
   At n=1000, Recall@10 was 1.000 at every ef on every distribution. A column of
   identical 1.000s means the dataset is too easy for approximation error to
   appear at all -- there is no trade-off being measured. Separation between ef
   values is the signal that the benchmark has become meaningful.

3. How does build cost grow?
   Every insert runs a full ef_construction search against a graph that is
   itself growing, so build time should be superlinear. The measured exponent
   decides whether 100k is reachable in pure Python or whether the next step has
   to be a different implementation strategy.

Scope and honesty
-----------------
Nothing here is multi-threaded, batched, or SIMD-optimised beyond what NumPy
does internally. These numbers describe *this implementation* of HNSW in
CPython, not HNSW the algorithm. A C++ implementation would move the crossover
dramatically to the left; the shape of the curves is the transferable result,
not their absolute position.

Results are written to benchmarks/results/scaling_<tag>.csv so a later run
(--tag optimised) can be diffed against this one rather than compared from
memory.
"""

import argparse
import csv
import gc
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from app.hnsw import HNSWIndex
from app.index import FlatIndex
from app.ivfpq import IVFPQIndex

K = 10
EF_VALUES = (25, 100)
NPROBE_VALUES = (4, 16)
PQ_M = 8
PROGRESS_EVERY = 2500
PROGRESS_THRESHOLD = 10000

RESULTS_DIR = Path("benchmarks") / "results"


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolating: these are latency samples, and reporting a
    value that was actually observed is more honest than one synthesised
    between two neighbours.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(pct / 100.0 * (len(ordered) - 1) + 0.5)
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def time_queries(
    search_fn: Callable[[np.ndarray], List[Tuple[str, float]]],
    queries: np.ndarray,
) -> Tuple[List[float], List[Set[str]]]:
    """Run every query once, returning per-query latency (ms) and result ids.

    Latency and results come from the same pass so recall and timing always
    describe the identical set of calls.
    """
    latencies: List[float] = []
    results: List[Set[str]] = []
    for row in queries:
        start = time.perf_counter()
        hits = search_fn(row)
        latencies.append((time.perf_counter() - start) * 1000.0)
        results.append({vid for vid, _ in hits})
    return latencies, results


def recall_at_k(results: Sequence[Set[str]], truth: Sequence[Set[str]]) -> float:
    if not truth:
        return float("nan")
    return sum(len(r & t) / K for r, t in zip(results, truth)) / len(truth)


def qps_from(latencies: Sequence[float]) -> float:
    """Throughput implied by mean single-threaded latency.

    Not a concurrency measurement -- there is no concurrency here. It is the
    mean latency expressed the other way up, which is easier to compare across
    rows than milliseconds.
    """
    if not latencies:
        return float("nan")
    mean_ms = sum(latencies) / len(latencies)
    return 1000.0 / mean_ms if mean_ms > 0 else float("inf")


def hnsw_edge_count(index: HNSWIndex) -> int:
    return sum(
        len(neighbours)
        for layer in index.neighbors.values()
        for neighbours in layer.values()
    )


# ---------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------- #


def build_clustered(
    n: int,
    dim: int,
    n_clusters: int = 50,
    sigma: float = 0.06,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """Clustered synthetic vectors.

    Clustered rather than uniform because uniform vectors in high dimensions are
    near-equidistant -- every point is roughly as far from the query as every
    other, so recall saturates at 1.000 and the benchmark measures nothing.

    Cluster count scales with n so that points-per-cluster stays roughly
    constant. Without this, a larger N is simply a denser blob and the task gets
    easier as N grows, which would confound exactly the comparison this script
    exists to make.
    """
    n_clusters = max(10, min(n_clusters, n // 20))
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.0, 1.0, (n_clusters, dim))

    # Integer division leaves a remainder; spread it over the first clusters so
    # the result is exactly n rows rather than n rounded down.
    per = n // n_clusters
    remainder = n - per * n_clusters
    counts = [per + (1 if i < remainder else 0) for i in range(n_clusters)]

    blocks = [
        centers[i] + rng.normal(0.0, sigma, (count, dim))
        for i, count in enumerate(counts)
        if count > 0
    ]
    vectors = np.vstack(blocks).astype(np.float32)
    ids = [f"v{i}" for i in range(n)]
    return ids, vectors


# ---------------------------------------------------------------------- #
# per-size measurement
# ---------------------------------------------------------------------- #


def measure_size(
    n: int,
    dim: int,
    n_queries: int,
    measure_memory: bool,
) -> List[Dict[str, object]]:
    """Build all three indexes at this N and measure recall, latency, memory."""
    ids, vectors = build_clustered(n, dim)

    # Evenly spaced sample rather than random: the same positions are queried at
    # every N, so rows stay comparable down a column.
    q_count = min(n_queries, n)
    q_idx = np.linspace(0, n - 1, q_count).astype(int)
    queries = vectors[q_idx]

    rows: List[Dict[str, object]] = []
    print(f"\n=== n={n} (dim={dim}, {q_count} queries, k={K}) ===")

    # ---------------- Flat: exact, and the ground truth for recall -------- #
    gc.collect()
    if measure_memory:
        tracemalloc.start()
    start = time.perf_counter()
    flat = FlatIndex(dim=dim)
    for vid, vec in zip(ids, vectors):
        flat.insert(vid, vec)
    flat_build = time.perf_counter() - start
    if measure_memory:
        flat_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        flat_mem = n * dim * 4

    flat_lat, truth = time_queries(
        lambda row: flat.search(row, k=K, metric="cosine"), queries
    )
    print(f"  flat      build {flat_build:7.2f}s  p50 {percentile(flat_lat, 50):7.3f}ms")
    rows.append(
        {
            "n": n,
            "index": "flat",
            "param": "exact",
            "recall": 1.0,
            "p50": percentile(flat_lat, 50),
            "p99": percentile(flat_lat, 99),
            "qps": qps_from(flat_lat),
            "build_s": flat_build,
            "memory_mb": flat_mem / (1024 * 1024),
        }
    )

    # ---------------- HNSW ------------------------------------------------ #
    gc.collect()
    if measure_memory:
        tracemalloc.start()
    start = time.perf_counter()
    hnsw = HNSWIndex(dim=dim, metric="cosine")
    for i, (vid, vec) in enumerate(zip(ids, vectors), start=1):
        hnsw.insert(vid, vec)
        # A 10k build takes many minutes. Without this it is indistinguishable
        # from a hang, and someone will kill it.
        if n >= PROGRESS_THRESHOLD and i % PROGRESS_EVERY == 0:
            print(
                f"    ... hnsw {i}/{n} inserted "
                f"({time.perf_counter() - start:.1f}s elapsed)"
            )
    hnsw_build = time.perf_counter() - start
    if measure_memory:
        hnsw_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        # Vectors at full precision, plus the graph itself. 8 bytes per directed
        # edge is a rough pointer/reference cost, not a measured figure.
        hnsw_mem = n * dim * 4 + 8 * hnsw_edge_count(hnsw)

    for ef in EF_VALUES:
        lat, res = time_queries(lambda row, e=ef: hnsw.search(row, k=K, ef=e), queries)
        rec = recall_at_k(res, truth)
        print(
            f"  hnsw ef={ef:<4} build {hnsw_build:7.2f}s  "
            f"p50 {percentile(lat, 50):7.3f}ms  recall {rec:.3f}"
        )
        rows.append(
            {
                "n": n,
                "index": "hnsw",
                "param": f"ef={ef}",
                "recall": rec,
                "p50": percentile(lat, 50),
                "p99": percentile(lat, 99),
                "qps": qps_from(lat),
                "build_s": hnsw_build,
                "memory_mb": hnsw_mem / (1024 * 1024),
            }
        )

    # ---------------- IVF+PQ ---------------------------------------------- #
    # sqrt(N) is the standard starting point for nlist: it balances the cost of
    # scanning centroids against the number of vectors per cell.
    nlist = max(8, min(int(n ** 0.5), 1024))

    gc.collect()
    if measure_memory:
        tracemalloc.start()
    start = time.perf_counter()
    ivf = IVFPQIndex(dim=dim, nlist=nlist, m=PQ_M, metric="cosine", seed=0)
    ivf.train(vectors)
    for vid, vec in zip(ids, vectors):
        ivf.add(vid, vec)
    ivf_build = time.perf_counter() - start
    if measure_memory:
        ivf_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        ivf_mem = ivf.memory_bytes()["total"]

    for nprobe in NPROBE_VALUES:
        lat, res = time_queries(
            lambda row, p=nprobe: ivf.search(row, k=K, nprobe=p), queries
        )
        rec = recall_at_k(res, truth)
        print(
            f"  ivfpq np={nprobe:<3} build {ivf_build:7.2f}s  "
            f"p50 {percentile(lat, 50):7.3f}ms  recall {rec:.3f}  (nlist={nlist})"
        )
        rows.append(
            {
                "n": n,
                "index": "ivfpq",
                "param": f"nprobe={nprobe}",
                "recall": rec,
                "p50": percentile(lat, 50),
                "p99": percentile(lat, 99),
                "qps": qps_from(lat),
                "build_s": ivf_build,
                "memory_mb": ivf_mem / (1024 * 1024),
            }
        )

    return rows


# ---------------------------------------------------------------------- #
# reporting
# ---------------------------------------------------------------------- #


def write_csv(rows: Sequence[Dict[str, object]], tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"scaling_{tag}.csv"
    fields = ["n", "index", "param", "recall", "p50", "p99", "qps", "build_s", "memory_mb"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    header = (
        f"{'N':>7} | {'index':<6} | {'param':<11} | {'recall':>6} | "
        f"{'p50 ms':>8} | {'p99 ms':>8} | {'QPS':>9} | {'build s':>8} | {'mem MB':>7}"
    )
    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:>7} | {r['index']:<6} | {r['param']:<11} | "
            f"{r['recall']:>6.3f} | {r['p50']:>8.3f} | {r['p99']:>8.3f} | "
            f"{r['qps']:>9.1f} | {r['build_s']:>8.2f} | {r['memory_mb']:>7.2f}"
        )


def print_crossover(rows: Sequence[Dict[str, object]], sizes: Sequence[int]) -> None:
    """Compare flat against HNSW ef=100 at every N and locate the crossover."""
    print("\n" + "=" * 72)
    print("CROSSOVER: flat p50 vs hnsw ef=100 p50")
    print("=" * 72)

    first_win: int = 0
    for n in sizes:
        flat_row = next(
            (r for r in rows if r["n"] == n and r["index"] == "flat"), None
        )
        hnsw_row = next(
            (
                r
                for r in rows
                if r["n"] == n and r["index"] == "hnsw" and r["param"] == "ef=100"
            ),
            None,
        )
        if flat_row is None or hnsw_row is None:
            print(f"  n={n:>7}: incomplete data, skipped")
            continue

        flat_p50 = float(flat_row["p50"])
        hnsw_p50 = float(hnsw_row["p50"])
        if hnsw_p50 < flat_p50:
            ratio = flat_p50 / hnsw_p50 if hnsw_p50 > 0 else float("inf")
            print(
                f"  n={n:>7}: HNSW faster by {ratio:5.2f}x  "
                f"(flat {flat_p50:.3f}ms vs hnsw {hnsw_p50:.3f}ms)"
            )
            if first_win == 0:
                first_win = n
        else:
            ratio = hnsw_p50 / flat_p50 if flat_p50 > 0 else float("inf")
            print(
                f"  flat faster by {ratio:5.2f}x at n={n:<7}  "
                f"(flat {flat_p50:.3f}ms vs hnsw {hnsw_p50:.3f}ms)"
            )

    print()
    if first_win:
        print(f"  -> crossover at n={first_win}: the first size where HNSW wins.")
    else:
        # Stating this explicitly matters. Silently omitting the section would
        # read as "not measured" rather than "measured, and the answer is no".
        print("  -> no crossover in this range: brute force still wins at every N")
        print("     tested. Either extend --sizes upward, or reduce the per-node")
        print("     Python overhead that dominates HNSW traversal at this scale.")


# ---------------------------------------------------------------------- #
# entry point
# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaling benchmark for Flat, HNSW and IVF+PQ."
    )
    parser.add_argument(
        "--sizes",
        default="1000,2500,5000,10000",
        help="comma-separated N values (default: 1000,2500,5000,10000)",
    )
    parser.add_argument("--dim", type=int, default=64, help="vector dimension")
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="queries per measurement; p99 needs ~100 samples to mean anything",
    )
    parser.add_argument("--tag", default="baseline", help="label for the output CSV")
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="use tracemalloc for real allocation figures (roughly doubles build time)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    # Printed before any work so an empty-file or import failure is obvious
    # immediately rather than after a long silent build.
    print("=" * 72)
    print("nanovec scaling benchmark")
    print("=" * 72)
    print(f"  sizes          : {sizes}")
    print(f"  dim            : {args.dim}")
    print(f"  queries        : {args.queries}")
    print(f"  k              : {K}")
    print(f"  ef values      : {list(EF_VALUES)}")
    print(f"  nprobe values  : {list(NPROBE_VALUES)}")
    print(f"  tag            : {args.tag}")
    print(f"  memory         : {'tracemalloc' if args.measure_memory else 'analytic estimate'}")
    if not args.measure_memory:
        print("                   (payload estimate only, not measured allocation;")
        print("                    pass --measure-memory for real figures)")
    print("=" * 72)

    if args.dim % PQ_M != 0:
        raise SystemExit(
            f"--dim must be divisible by {PQ_M} for product quantization "
            f"(got {args.dim})"
        )

    rows: List[Dict[str, object]] = []
    for n in sizes:
        rows.extend(measure_size(n, args.dim, args.queries, args.measure_memory))

    print_summary(rows)
    print_crossover(rows, sizes)

    path = write_csv(rows, args.tag)
    print(f"\nwrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
