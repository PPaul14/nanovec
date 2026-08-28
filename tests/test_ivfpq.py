"""IVF+PQ tests.

Note the recall thresholds are much lower than HNSW's. That is expected, not a
defect: PQ throws away information at encode time, so it can never reach 1.000
the way a well-built graph can. The trade is memory, and these tests pin down
both sides of it.
"""

import numpy as np
import pytest

from app.index import FlatIndex
from app.ivfpq import IVFPQIndex, kmeans

DIM = 32


def _clustered(n_clusters=8, per_cluster=60, dim=DIM, sigma=0.08, seed=5):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, (n_clusters, dim))
    vectors = {}
    for ci, c in enumerate(centers):
        for j in range(per_cluster):
            vectors[f"c{ci}_v{j}"] = (c + rng.normal(0, sigma, dim)).tolist()
    return vectors


def _build(vectors, nlist=16, m=8, nprobe_train=True):
    idx = IVFPQIndex(dim=DIM, nlist=nlist, m=m, metric="cosine", seed=0)
    matrix = np.array(list(vectors.values()), dtype=np.float32)
    idx.train(matrix)
    for vid, vec in vectors.items():
        idx.add(vid, vec)
    return idx


# ---------------------------------------------------------------- k-means


def test_kmeans_returns_requested_centroid_count():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 8)).astype(np.float32)
    C = kmeans(X, k=10, seed=0)
    assert C.shape == (10, 8)


def test_kmeans_recovers_well_separated_clusters():
    rng = np.random.default_rng(1)
    true_centers = np.array([[-5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]], dtype=np.float32)
    X = np.vstack([c + rng.normal(0, 0.2, (50, 2)) for c in true_centers]).astype(np.float32)

    C = kmeans(X, k=3, seed=0)

    # Every true center should have a learned centroid sitting close to it.
    for tc in true_centers:
        assert np.min(np.linalg.norm(C - tc, axis=1)) < 1.0


def test_kmeans_handles_fewer_points_than_clusters():
    X = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    C = kmeans(X, k=5, seed=0)
    assert C.shape == (5, 2)
    assert not np.isnan(C).any()


# ---------------------------------------------------------------- guards


def test_search_before_train_raises():
    idx = IVFPQIndex(dim=DIM, nlist=4)
    with pytest.raises(RuntimeError):
        idx.search([0.1] * DIM, k=3)


def test_add_before_train_raises():
    idx = IVFPQIndex(dim=DIM, nlist=4)
    with pytest.raises(RuntimeError):
        idx.add("a", [0.1] * DIM)


def test_m_must_divide_dim():
    with pytest.raises(ValueError):
        IVFPQIndex(dim=30, m=8)


def test_train_requires_enough_vectors():
    idx = IVFPQIndex(dim=DIM, nlist=64)
    with pytest.raises(ValueError):
        idx.train(np.zeros((10, DIM), dtype=np.float32))


# ---------------------------------------------------------------- recall


def test_recall_against_flat_baseline():
    vectors = _clustered()
    idx = _build(vectors)

    flat = FlatIndex(dim=DIM)
    for vid, vec in vectors.items():
        flat.insert(vid, vec)

    query_ids = list(vectors.keys())[::17][:20]
    total = 0.0
    for qid in query_ids:
        truth = {i for i, _ in flat.search(vectors[qid], k=10, metric="cosine")}
        approx = {i for i, _ in idx.search(vectors[qid], k=10, nprobe=8)}
        total += len(truth & approx) / 10

    recall = total / len(query_ids)
    print(f"\nIVF+PQ Recall@10 (nprobe=8): {recall:.3f}")
    # PQ is lossy by construction; this is a floor, not a target.
    assert recall >= 0.40, f"recall too low: {recall}"


def test_recall_improves_with_nprobe():
    """More probed cells = more candidates = better recall. The core IVF knob."""
    vectors = _clustered()
    idx = _build(vectors, nlist=16)

    flat = FlatIndex(dim=DIM)
    for vid, vec in vectors.items():
        flat.insert(vid, vec)

    query_ids = list(vectors.keys())[::23][:15]

    def recall_at(nprobe):
        total = 0.0
        for qid in query_ids:
            truth = {i for i, _ in flat.search(vectors[qid], k=10, metric="cosine")}
            approx = {i for i, _ in idx.search(vectors[qid], k=10, nprobe=nprobe)}
            total += len(truth & approx) / 10
        return total / len(query_ids)

    r1, r16 = recall_at(1), recall_at(16)
    print(f"\nnprobe=1: {r1:.3f}   nprobe=16: {r16:.3f}")
    assert r16 >= r1, "recall should not degrade as more cells are probed"


def test_self_query_returns_self_in_top_k():
    """A stored vector should find itself. PQ error can move it off rank 1,
    but it should still be in the top few."""
    vectors = _clustered()
    idx = _build(vectors)
    hits = 0
    probes = list(vectors.keys())[::13][:20]
    for qid in probes:
        ids = [i for i, _ in idx.search(vectors[qid], k=5, nprobe=16)]
        if qid in ids:
            hits += 1
    assert hits / len(probes) >= 0.7


# ---------------------------------------------------------------- memory


def test_compression_ratio_matches_expectation():
    idx = IVFPQIndex(dim=128, nlist=16, m=8)
    # 128 dims * 4 bytes = 512 bytes raw, vs 8 bytes of codes.
    assert idx.compression_ratio() == 64.0


def test_memory_accounting_is_consistent():
    vectors = _clustered()
    idx = _build(vectors)
    mem = idx.memory_bytes()

    n = len(vectors)
    assert mem["codes"] == n * idx.m
    assert mem["raw_float32_equivalent"] == n * DIM * 4
    assert mem["total"] == mem["codes"] + mem["codebooks"] + mem["coarse_centroids"]
    print(
        f"\ncodes={mem['codes']}B  codebooks={mem['codebooks']}B  "
        f"raw would be {mem['raw_float32_equivalent']}B"
    )


# ---------------------------------------------------------------- CRUD


def test_delete_excludes_from_results():
    vectors = _clustered()
    idx = _build(vectors)
    target = "c0_v0"
    idx.delete(target)
    ids = [i for i, _ in idx.search(vectors[target], k=10, nprobe=16)]
    assert target not in ids
    assert idx.live_count == len(vectors) - 1


def test_delete_unknown_raises():
    vectors = _clustered()
    idx = _build(vectors)
    with pytest.raises(KeyError):
        idx.delete("nope")


def test_duplicate_add_rejected():
    vectors = _clustered()
    idx = _build(vectors)
    with pytest.raises(ValueError):
        idx.add("c0_v0", vectors["c0_v0"])


def test_metadata_filter():
    rng = np.random.default_rng(3)
    vectors = {f"v{i}": rng.uniform(-1, 1, DIM).tolist() for i in range(200)}
    idx = IVFPQIndex(dim=DIM, nlist=8, m=8, seed=0)
    idx.train(np.array(list(vectors.values()), dtype=np.float32))
    for i, (vid, vec) in enumerate(vectors.items()):
        idx.add(vid, vec, {"tier": "gold" if i % 5 == 0 else "silver"})

    results = idx.search(vectors["v0"], k=5, nprobe=8, filter={"tier": "gold"})
    assert results
    for vid, _ in results:
        assert idx.metadata[vid]["tier"] == "gold"


def test_empty_index_search_returns_empty():
    idx = IVFPQIndex(dim=DIM, nlist=8, m=8)
    idx.train(np.random.default_rng(0).uniform(-1, 1, (50, DIM)).astype(np.float32))
    assert idx.search([0.1] * DIM, k=5) == []