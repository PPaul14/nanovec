"""
Is FlatIndex.insert amortised O(1), or does it copy the world on every call?

Run from the project root:
    venv/Scripts/python.exe -m benchmarks.insert_scaling
    venv/Scripts/python.exe -m benchmarks.insert_scaling --sizes 1000,2000,4000

Why this benchmark exists
-------------------------
The 100k scaling run showed FlatIndex taking 683s to load 100,000 vectors
against 2.7s for 10,000 -- roughly 250x the time for 10x the data. Nothing
about a brute-force index should behave that way; the per-vector work is a
single array write.

The cause was `insert` doing `self.vectors = np.vstack([self.vectors, vec])`.
`vstack` allocates a new array and copies every existing row, so inserting N
vectors copies about N^2/2 rows in total. At n=1000 that is invisible. At
n=100000 it is the entire runtime.

This benchmark is the measurement that caught it, kept so the property can be
re-checked rather than assumed. It doubles N at each step and reports the ratio
between consecutive rows:

    ratio ~2x  -> total work is linear, i.e. amortised O(1) per insert
    ratio ~4x  -> total work is quadratic, i.e. an O(N) copy per insert

The fitted exponent is the same statement as a single number: t ~ N^a, where
a = 1 is linear and a = 2 is quadratic.

Measured on this machine, dim=64:

    N        before (vstack)   after (capacity doubling)
    1000       0.022s              0.002s
    2000       0.069s              0.005s
    4000       0.816s              0.008s
    8000       6.053s              0.016s
    16000     26.333s              0.031s
    exponent    2.56                1.02

The `us/insert` column is the clearest signal: constant for an amortised O(1)
append, rising with N for a quadratic one. Before the fix it went from 22us to
1646us; after, it sits flat near 2us.
"""

import argparse
import time
from typing import List, Sequence, Tuple

import numpy as np

from app.index import FlatIndex

DIM = 64
DEFAULT_SIZES = (1000, 2000, 4000, 8000, 16000)
SEED = 42


def build_vectors(n: int, dim: int, seed: int = SEED) -> np.ndarray:
    """Uniform random vectors.

    Distribution is irrelevant here -- this measures allocation and copying,
    which do not care what the numbers are. Uniform keeps generation cheap so
    the setup does not pollute the timing.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, dim)).astype(np.float32)


def time_loop_insert(vectors: np.ndarray) -> float:
    index = FlatIndex(dim=vectors.shape[1])
    start = time.perf_counter()
    for i, vec in enumerate(vectors):
        index.insert(f"v{i}", vec)
    return time.perf_counter() - start


def time_batch_insert(vectors: np.ndarray) -> float:
    index = FlatIndex(dim=vectors.shape[1])
    ids = [f"v{i}" for i in range(vectors.shape[0])]
    start = time.perf_counter()
    index.insert_many(ids, vectors)
    return time.perf_counter() - start


def fitted_exponents(results: Sequence[Tuple[int, float]]) -> List[float]:
    """t ~ N^a between each consecutive pair. a=1 linear, a=2 quadratic."""
    out = []
    for i in range(1, len(results)):
        (n0, t0), (n1, t1) = results[i - 1], results[i]
        if t0 > 0 and n0 > 0:
            out.append(float(np.log(t1 / t0) / np.log(n1 / n0)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether FlatIndex.insert is amortised O(1)."
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="comma-separated N values; each should double the previous",
    )
    parser.add_argument("--dim", type=int, default=DIM)
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"FlatIndex insert scaling, dim={args.dim}\n")
    header = (
        f"{'N':>7} | {'insert() s':>10} | {'us/insert':>10} | {'ratio':>7} | "
        f"{'insert_many() s':>15} | {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    loop_results: List[Tuple[int, float]] = []
    prev = None
    for n in sizes:
        vectors = build_vectors(n, args.dim)
        t_loop = time_loop_insert(vectors)
        t_batch = time_batch_insert(vectors)

        ratio = "-" if prev is None or prev == 0 else f"{t_loop / prev:.2f}x"
        speedup = "-" if t_batch <= 0 else f"{t_loop / t_batch:.1f}x"
        print(
            f"{n:>7} | {t_loop:>10.3f} | {1e6 * t_loop / n:>10.1f} | {ratio:>7} | "
            f"{t_batch:>15.4f} | {speedup:>8}"
        )
        loop_results.append((n, t_loop))
        prev = t_loop

    print()
    print("Each row doubles N.")
    print("  ratio ~2x -> linear total work  (amortised O(1) per insert)")
    print("  ratio ~4x -> quadratic total work (an O(N) copy per insert)")

    exps = fitted_exponents(loop_results)
    if exps:
        mean = float(np.mean(exps))
        print(f"\nfitted exponent per step: {[f'{e:.2f}' for e in exps]}")
        print(f"mean exponent: {mean:.2f}   (1.0 = linear, 2.0 = quadratic)")
        verdict = "amortised O(1)" if mean < 1.4 else "SUPERLINEAR -- regression?"
        print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
