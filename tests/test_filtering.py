"""Metadata filtering + tombstone delete tests.

The interesting case is a SELECTIVE filter -- one that matches only a small
fraction of the data. That is where pre- and post-filtering diverge sharply,
and it is the scenario real workloads hit (e.g. "nearest neighbours, but only
in stock, only this tenant").
"""

import numpy as np
import pytest

from app.hnsw import HNSWIndex

DIM = 16


def _build(n=400, rare_every=40, seed=3):
    """n vectors; 1 in `rare_every` tagged category='rare'."""
    rng = np.random.default_rng(seed)
    idx = HNSWIndex(dim=DIM, metric="cosine")
    vectors = {}
    for i in range(n):
        v = rng.uniform(-1, 1, DIM).tolist()
        vid = f"v{i}"
        vectors[vid] = v
        category = "rare" if i % rare_every == 0 else "common"
        idx.insert(vid, v, {"category": category, "idx": i})
    return idx, vectors


def test_filter_returns_only_matching_metadata():
    idx, vectors = _build()
    results = idx.search(vectors["v0"], k=5, filter={"category": "rare"})
    assert results
    for vid, _ in results:
        assert idx.metadata[vid]["category"] == "rare"


def test_pre_filter_returns_full_k_on_selective_filter():
    """Pre-filtering keeps exploring until it has k matches."""
    idx, vectors = _build()
    results = idx.search(vectors["v7"], k=5, filter={"category": "rare"}, filter_mode="pre")
    assert len(results) == 5


def test_post_filter_may_return_fewer_than_k():
    """Post-filtering searches first and filters after, so a selective filter
    can leave fewer than k survivors. This documents the trade-off rather than
    asserting a specific shortfall."""
    idx, vectors = _build()
    pre = idx.search(vectors["v7"], k=5, filter={"category": "rare"}, filter_mode="pre")
    post = idx.search(vectors["v7"], k=5, filter={"category": "rare"}, filter_mode="post")

    assert all(idx.metadata[i]["category"] == "rare" for i, _ in post)
    assert len(post) <= len(pre)


def test_multi_key_filter_is_conjunctive():
    idx = HNSWIndex(dim=DIM)
    rng = np.random.default_rng(0)
    for i in range(30):
        idx.insert(
            f"v{i}",
            rng.uniform(-1, 1, DIM).tolist(),
            {"tenant": "a" if i < 15 else "b", "active": i % 2 == 0},
        )

    results = idx.search([0.5] * DIM, k=10, filter={"tenant": "a", "active": True})
    for vid, _ in results:
        meta = idx.metadata[vid]
        assert meta["tenant"] == "a" and meta["active"] is True


def test_filter_matching_nothing_returns_empty():
    idx, vectors = _build(n=50)
    assert idx.search(vectors["v0"], k=5, filter={"category": "nonexistent"}) == []


def test_unfiltered_search_unaffected():
    """Regression guard: adding filtering must not change unfiltered behaviour."""
    idx, vectors = _build(n=100)
    results = idx.search(vectors["v10"], k=5)
    assert len(results) == 5
    assert results[0][0] == "v10"  # a vector is its own nearest neighbour


def test_deleted_vectors_excluded_from_results():
    idx, vectors = _build(n=100)
    target = idx.search(vectors["v10"], k=3)
    assert target[0][0] == "v10"

    idx.delete("v10")
    after = idx.search(vectors["v10"], k=3)
    assert "v10" not in {i for i, _ in after}
    assert len(after) == 3  # graph still navigable through the tombstone


def test_delete_is_idempotent_and_updates_count():
    idx, _ = _build(n=20)
    assert idx.live_count == 20
    idx.delete("v1")
    idx.delete("v1")
    assert idx.live_count == 19
    assert len(idx.deleted) == 1


def test_delete_unknown_id_raises():
    idx, _ = _build(n=10)
    with pytest.raises(KeyError):
        idx.delete("does-not-exist")


def test_reinsert_after_delete_resurrects():
    idx, vectors = _build(n=30)
    idx.delete("v5")
    assert idx.live_count == 29

    idx.insert("v5", vectors["v5"], {"category": "resurrected"})
    assert idx.live_count == 30
    assert idx.metadata["v5"]["category"] == "resurrected"
    assert "v5" in {i for i, _ in idx.search(vectors["v5"], k=3)}


def test_invalid_filter_mode_rejected():
    idx, vectors = _build(n=10)
    with pytest.raises(ValueError):
        idx.search(vectors["v0"], k=3, filter_mode="sideways")