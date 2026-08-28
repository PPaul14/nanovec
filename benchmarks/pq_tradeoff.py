"""
What actually limits IVF+PQ recall: cell coverage, or quantization error?

Run from the project root:
    venv/Scripts/python.exe -m benchmarks.pq_tradeoff

The question this answers
-------------------------
In benchmarks/compare_indexes.py, IVF+PQ recall is identical from nprobe=4
through nprobe=16. That is suspicious: raising nprobe scans strictly more cells,
so if coverage were the binding constraint recall would keep climbing.

This script isolates the two candidate causes.

  Coverage        -> probe every cell (nprobe = nlist). If recall still does not
                     move, no vector is being missed for coverage reasons and the
                     coarse quantizer is not the limit.
  Quantization    -> sweep m, the number of PQ subspaces. Larger m means shorter
                     sub-vectors, so each 256-entry codebook describes a smaller
                     slice of the space and reconstruction error falls. If recall
                     tracks m, the codebooks are the limit.

Only one of those can be true, and the sweep below settles it.

The trade-off is the point
--------------------------
Recall bought this way is not free: a vector costs m bytes, so compression is
dim*4/m. Every step up in recall is a step down in the compression that is the
entire reason to choose IVF+PQ over HNSW. The curve, not any single row, is the
result.

Dataset is imported from compare_indexes so both benchmarks provably measure the
same vectors -- the numbers here line up with that table rather than being a
separate experiment that happens to look similar.
"""

import numpy as np

from app.index import FlatIndex
from app.ivfpq import IVFPQIndex
from benchmarks.compare_indexes import DIM, K, N_QUERIES, build_clustered

NLIST = 32
M_VALUES = (4, 8, 16, 32)
LOW_NPROBE = 1
SEED = 0


def ground_truth(vectors, query_ids):
    """Exact top-K from the brute-force index, used as the recall baseline."""
    flat = FlatIndex(dim=DIM)
    for vid, vec in vectors.items():
        flat.insert(vid, vec)
    return {
        q: {i for i, _ in flat.search(vectors[q], k=K, metric="cosine")}
        for q in query_ids
    }


def recall_at(index, vectors, query_ids, truth, nprobe):
    total = 0.0
    for q in query_ids:
        got = {i for i, _ in index.search(vectors[q], k=K, nprobe=nprobe)}
        total += len(truth[q] & got) / K
    return total / len(query_ids)


def main():
    vectors = build_clustered()
    ids = list(vectors.keys())
    query_ids = ids[::67][:N_QUERIES]
    matrix = np.array(list(vectors.values()), dtype=np.float32)

    truth = ground_truth(vectors, query_ids)

    print(
        f"dataset: {len(vectors)} vectors, dim={DIM}, {N_QUERIES} queries, k={K}"
    )
    print(f"nlist={NLIST}  ->  nprobe={NLIST} scans every cell (no coverage loss)\n")

    header = (
        f"{'m':>3} | {'subvec dim':>10} | {'compression':>11} | "
        f"{'nprobe=' + str(LOW_NPROBE):>9} | {'nprobe=' + str(NLIST) + ' (all)':>15}"
    )
    print(header)
    print("-" * len(header))

    for m in M_VALUES:
        index = IVFPQIndex(dim=DIM, nlist=NLIST, m=m, metric="cosine", seed=SEED)
        index.train(matrix)
        for vid, vec in vectors.items():
            index.add(vid, vec)

        low = recall_at(index, vectors, query_ids, truth, LOW_NPROBE)
        full = recall_at(index, vectors, query_ids, truth, NLIST)

        print(
            f"{m:>3} | {DIM // m:>9}d | {index.compression_ratio():>10.0f}x | "
            f"{low:>9.3f} | {full:>15.3f}"
        )

    print(
        "\nReading this table:\n"
        "  Down a column -> recall rises with m, so PQ reconstruction error is\n"
        "                   what limits recall.\n"
        "  Across a row   -> probing every cell instead of one adds little, so\n"
        "                   cell coverage is not what limits recall.\n"
        "  Compression    -> falls as m rises. Recall bought here is paid for in\n"
        "                   the memory saving that motivates IVF+PQ at all.\n"
    )
    print(
        "Caveat: with 2000 training vectors and ksub=256, each codebook centroid\n"
        "sees roughly 8 points. That is far below what a 256-entry codebook needs,\n"
        "so these figures are provisional until the Week 5 scale-up.\n"
    )


if __name__ == "__main__":
    main()
