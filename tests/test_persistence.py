"""Durability tests: snapshot round-trip, WAL replay, crash recovery."""

import json

import numpy as np
import pytest

from app.hnsw import HNSWIndex
from app.persistence import (
    WAL_FILE,
    WriteAheadLog,
    load_snapshot,
    recover,
    save_snapshot,
)

DIM = 16


def _vec(seed: int):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, DIM).tolist()


def _build(n=50, seed=0):
    idx = HNSWIndex(dim=DIM, metric="cosine")
    vectors = {}
    for i in range(n):
        v = _vec(seed + i)
        vid = f"v{i}"
        vectors[vid] = v
        idx.insert(vid, v, {"group": "even" if i % 2 == 0 else "odd"})
    return idx, vectors


def test_snapshot_roundtrip_preserves_search_results(tmp_path):
    idx, vectors = _build()
    before = idx.search(vectors["v7"], k=5)

    save_snapshot(idx, tmp_path)
    restored = load_snapshot(tmp_path)

    assert restored is not None
    after = restored.search(vectors["v7"], k=5)

    # Same ids in the same order, and scores match to float32 precision.
    assert [i for i, _ in before] == [i for i, _ in after]
    for (_, s1), (_, s2) in zip(before, after):
        assert abs(s1 - s2) < 1e-6


def test_snapshot_preserves_graph_and_tombstones(tmp_path):
    idx, _ = _build()
    idx.delete("v3")
    idx.delete("v4")

    save_snapshot(idx, tmp_path)
    restored = load_snapshot(tmp_path)

    assert restored.deleted == {"v3", "v4"}
    assert restored.entry_point == idx.entry_point
    assert restored.max_level == idx.max_level
    # Layer keys must come back as ints, not the strings JSON stored them as.
    assert all(isinstance(layer, int) for layer in restored.neighbors)
    assert restored.neighbors[0] == idx.neighbors[0]


def test_load_snapshot_returns_none_when_absent(tmp_path):
    assert load_snapshot(tmp_path) is None


def test_wal_replay_reconstructs_index(tmp_path):
    """No snapshot at all -- the log alone must be enough to rebuild."""
    wal = WriteAheadLog(tmp_path / WAL_FILE)
    vectors = {}
    for i in range(20):
        v = _vec(100 + i)
        vectors[f"v{i}"] = v
        wal.append({"op": "insert", "id": f"v{i}", "vector": v, "metadata": {"i": i}})
    wal.append({"op": "delete", "id": "v5"})

    recovered = recover(tmp_path, dim=DIM)

    assert recovered.live_count == 19
    assert "v5" in recovered.deleted
    assert {i for i, _ in recovered.search(vectors["v0"], k=5)}.isdisjoint({"v5"})


def test_recovery_applies_wal_on_top_of_snapshot(tmp_path):
    idx, vectors = _build(n=30)
    save_snapshot(idx, tmp_path)

    # Mutations that happened after the snapshot was taken.
    wal = WriteAheadLog(tmp_path / WAL_FILE)
    new_vec = _vec(999)
    wal.append({"op": "insert", "id": "late", "vector": new_vec, "metadata": {"g": "x"}})
    wal.append({"op": "delete", "id": "v1"})

    recovered = recover(tmp_path, dim=DIM)

    assert "late" in recovered.data
    assert "v1" in recovered.deleted
    assert recovered.live_count == 30  # 30 inserted, +1 added, -1 deleted
    assert recovered.metadata["late"] == {"g": "x"}


def test_replay_is_idempotent(tmp_path):
    """Replaying a record already captured by the snapshot must not corrupt state."""
    idx, _ = _build(n=10)
    save_snapshot(idx, tmp_path)

    wal = WriteAheadLog(tmp_path / WAL_FILE)
    # Deliberately re-log an insert the snapshot already contains.
    wal.append({"op": "insert", "id": "v0", "vector": _vec(0), "metadata": None})

    recovered = recover(tmp_path, dim=DIM)
    assert recovered.live_count == 10


def test_torn_final_wal_record_is_ignored(tmp_path):
    """Simulate a crash mid-append: last line is truncated JSON."""
    wal_path = tmp_path / WAL_FILE
    wal = WriteAheadLog(wal_path)
    v = _vec(1)
    wal.append({"op": "insert", "id": "good", "vector": v, "metadata": None})

    with open(wal_path, "a", encoding="utf-8") as f:
        f.write('{"op": "insert", "id": "torn", "vec')  # no newline, no closing brace

    recovered = recover(tmp_path, dim=DIM)

    assert "good" in recovered.data
    assert "torn" not in recovered.data


def test_snapshot_is_atomic_no_tmp_files_left(tmp_path):
    idx, _ = _build(n=5)
    save_snapshot(idx, tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_index_snapshot_roundtrip(tmp_path):
    idx = HNSWIndex(dim=DIM)
    save_snapshot(idx, tmp_path)
    restored = load_snapshot(tmp_path)
    assert restored is not None
    assert restored.live_count == 0
    assert restored.search([0.0] * DIM, k=5) == []