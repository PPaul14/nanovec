"""
Durability layer: write-ahead log + snapshots.

The problem: HNSWIndex lives entirely in RAM. Kill the process and everything
is gone. Rebuilding the graph from scratch is expensive (every insert runs a
full ef_construction search), so we need both:

  WAL (write-ahead log)
      Every mutation is appended to an fsync'd JSONL file BEFORE it is applied
      in memory. If the process dies, replaying the log reconstructs the exact
      sequence of operations. Durable, but replay cost grows without bound.

  Snapshot
      A point-in-time dump of the whole index -- vectors, graph, metadata,
      tombstones. Loading one is far cheaper than replaying millions of WAL
      records, so taking a snapshot lets us truncate the log.

  Recovery = load latest snapshot, then replay whatever WAL records came after it.

This is the same shape as Postgres checkpoints + WAL, or Redis RDB + AOF.

On-disk layout (default `data/`):
    data/
      vectors.npy      float32 matrix, one row per id, in manifest["ids"] order
      manifest.json    graph structure, metadata, tombstones, index params
      wal.jsonl        append-only log of mutations since the last snapshot

Vectors are stored as .npy rather than JSON because float32 binary is ~5x
smaller and avoids decimal round-tripping. The graph goes to JSON because it is
small, and being human-readable makes debugging connectivity far easier.
"""

import json
import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from app.hnsw import HNSWIndex

SNAPSHOT_VECTORS = "vectors.npy"
SNAPSHOT_MANIFEST = "manifest.json"
WAL_FILE = "wal.jsonl"

PathLike = Union[str, Path]


class WriteAheadLog:
    """Append-only durable log of index mutations."""

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        """Append one record and force it to disk.

        The fsync is the entire point. Without it the write sits in the OS page
        cache and a power loss silently drops it -- which would make this a
        "write-behind log that usually works" rather than a WAL.
        """
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def replay(self) -> Iterator[dict]:
        """Yield records in order, stopping at the first corrupt line.

        A crash mid-append leaves a torn final line. That record never
        completed, so the client was never told it succeeded, and stopping
        there is the correct recovery behaviour.
        """
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    break

    def truncate(self) -> None:
        """Drop the log. Only safe immediately after a successful snapshot."""
        open(self.path, "w", encoding="utf-8").close()

    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


def save_snapshot(index: HNSWIndex, directory: PathLike) -> None:
    """Write a full point-in-time copy of the index.

    Both files are written to `.tmp` and then os.replace'd, which is atomic on
    POSIX and Windows. A crash mid-write therefore leaves the PREVIOUS snapshot
    intact instead of a half-written unusable one.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    ids = list(index.data.keys())
    if ids:
        matrix = np.vstack([index.data[i] for i in ids]).astype(np.float32)
    else:
        matrix = np.zeros((0, index.dim), dtype=np.float32)

    tmp_vectors = directory / (SNAPSHOT_VECTORS + ".tmp")
    with open(tmp_vectors, "wb") as f:
        np.save(f, matrix)

    manifest = {
        "dim": index.dim,
        "metric": index.metric,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "entry_point": index.entry_point,
        "max_level": index.max_level,
        "ids": ids,
        "levels": index.levels,
        "metadata": index.metadata,
        "deleted": sorted(index.deleted),
        # JSON object keys must be strings, so int layer numbers are stringified
        # here and converted back on load.
        "neighbors": {str(layer): nbrs for layer, nbrs in index.neighbors.items()},
    }

    tmp_manifest = directory / (SNAPSHOT_MANIFEST + ".tmp")
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_vectors, directory / SNAPSHOT_VECTORS)
    os.replace(tmp_manifest, directory / SNAPSHOT_MANIFEST)


def load_snapshot(directory: PathLike) -> Optional[HNSWIndex]:
    """Rebuild an index from a snapshot, or return None if there is not one."""
    directory = Path(directory)
    manifest_path = directory / SNAPSHOT_MANIFEST
    vectors_path = directory / SNAPSHOT_VECTORS
    if not manifest_path.exists() or not vectors_path.exists():
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(vectors_path, "rb") as f:
        matrix = np.load(f)

    index = HNSWIndex(
        dim=manifest["dim"],
        metric=manifest["metric"],
        M=manifest["M"],
        ef_construction=manifest["ef_construction"],
    )

    for row, vid in enumerate(manifest["ids"]):
        index.data[vid] = matrix[row].astype(np.float32)

    index.metadata = manifest["metadata"]
    index.levels = manifest["levels"]
    index.deleted = set(manifest["deleted"])
    index.entry_point = manifest["entry_point"]
    index.max_level = manifest["max_level"]
    index.neighbors = {int(layer): nbrs for layer, nbrs in manifest["neighbors"].items()}

    return index


def recover(directory: PathLike, dim: int, metric: str = "cosine") -> HNSWIndex:
    """Full recovery: snapshot (if any) + WAL replay on top."""
    directory = Path(directory)
    index = load_snapshot(directory)
    if index is None:
        index = HNSWIndex(dim=dim, metric=metric)

    wal = WriteAheadLog(directory / WAL_FILE)
    for record in wal.replay():
        op = record.get("op")
        try:
            if op == "insert":
                # Snapshot may already contain this record if the snapshot was
                # taken before the WAL was truncated. Applying it twice must be
                # harmless -- replay has to be idempotent.
                if record["id"] in index.data and record["id"] not in index.deleted:
                    continue
                index.insert(record["id"], record["vector"], record.get("metadata"))
            elif op == "delete":
                if record["id"] in index.data:
                    index.delete(record["id"])
        except (ValueError, KeyError):
            # A malformed record should not block recovery of the rest.
            continue

    return index