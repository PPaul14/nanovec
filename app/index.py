"""
Brute-force exact index.

This is the correctness ground truth every approximate index in the project is
measured against, so its results must not drift. It is also genuinely the
fastest option at small N: one `(N, dim) @ (dim,)` matmul is a single vectorised
C loop, which beats interpreted graph traversal until N gets large.

Storage note (Week 7)
---------------------
`insert` used to do `self.vectors = np.vstack([self.vectors, vec])`, which
allocates a whole new array and copies every existing row on EVERY call.
Inserting N vectors therefore copies ~N^2/2 rows in total -- quadratic work
hidden behind a one-line append. It is invisible at n=1000 and dominates
everything at n=100000.

The fix is the same one `HNSWIndex` already uses: keep a backing array with
spare capacity, track how many rows are live, and double the capacity when it
fills. Doubling makes the copies geometrically rarer, so the cost amortises to
O(1) per insert -- the same strategy behind `list.append`.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Small enough not to waste memory on a 3-dimensional test index, large enough
# that trivial workloads never reallocate at all.
_INITIAL_CAPACITY = 128


class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._id_to_pos: Dict[str, int] = {}

        # Allocated lazily on first insert, so `vectors` stays None on an
        # untouched index exactly as it did before.
        self._matrix: Optional[np.ndarray] = None
        self._count: int = 0

    # ------------------------------------------------------------------ #
    # storage
    # ------------------------------------------------------------------ #

    @property
    def vectors(self) -> Optional[np.ndarray]:
        """The live rows only -- never the padded tail.

        A view, not a copy, so `search` pays nothing for the indirection. Any
        read of stored vectors must go through here: touching `_matrix`
        directly would include uninitialised capacity in the results.
        """
        if self._matrix is None:
            return None
        return self._matrix[: self._count]

    def _ensure_capacity(self, needed: int) -> None:
        """Grow to hold `needed` rows, doubling so growth amortises to O(1)."""
        if self._matrix is None:
            capacity = _INITIAL_CAPACITY
            while capacity < needed:
                capacity *= 2
            self._matrix = np.zeros((capacity, self.dim), dtype=np.float32)
            return

        capacity = self._matrix.shape[0]
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        grown = np.zeros((capacity, self.dim), dtype=np.float32)
        grown[: self._count] = self._matrix[: self._count]
        self._matrix = grown

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #

    def insert(self, id: str, vector: List[float], metadata: Optional[dict] = None):
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
        if id in self._id_to_pos:
            raise ValueError(f"id '{id}' already exists")

        self._ensure_capacity(self._count + 1)
        self._matrix[self._count] = vec

        self.ids.append(id)
        self._id_to_pos[id] = self._count
        self.metadata[id] = metadata or {}
        self._count += 1

    def insert_many(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """Bulk load: one capacity grow and one block copy for the whole batch.

        Per-insert Python call overhead is paid N times by a loop over
        `insert`, which is pure waste when the caller already has the whole
        matrix in hand -- the case for every benchmark and bulk load here. This
        is the same reason FAISS exposes `add(matrix)` rather than only a
        single-vector entry point.

        Validation happens up front so a bad batch raises before anything is
        written, leaving the index unchanged rather than half-loaded.
        """
        block = np.asarray(vectors, dtype=np.float32)
        if block.ndim != 2:
            raise ValueError(f"expected a 2-D (n, dim) array, got shape {block.shape}")
        if block.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {block.shape[1]}")
        if len(ids) != block.shape[0]:
            raise ValueError(
                f"got {len(ids)} ids for {block.shape[0]} vectors -- they must match"
            )
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError(
                f"got {len(metadata)} metadata entries for {len(ids)} ids"
            )

        seen = set()
        for vid in ids:
            if vid in self._id_to_pos or vid in seen:
                raise ValueError(f"id '{vid}' already exists")
            seen.add(vid)

        n = block.shape[0]
        if n == 0:
            return

        self._ensure_capacity(self._count + n)
        self._matrix[self._count : self._count + n] = block

        for offset, vid in enumerate(ids):
            self.ids.append(vid)
            self._id_to_pos[vid] = self._count + offset
            self.metadata[vid] = (metadata[offset] if metadata else None) or {}
        self._count += n

    def delete(self, id: str):
        """Remove a vector, preserving the order of everything after it.

        Deliberately NOT swap-with-last. That would make this O(1) instead of
        O(N), but it reorders the backing rows, and `search` resolves ties by
        whatever order `np.argsort` sees -- so two vectors with identical scores
        could come back in a different order than before. This index is the
        ground truth for every recall number in the project, so its results are
        the one thing that must not drift for a performance win.

        The reallocation is still gone: rows shift down inside the existing
        buffer instead of `np.delete` building a new array each time.
        """
        if id not in self._id_to_pos:
            raise KeyError(f"id '{id}' not found")

        pos = self._id_to_pos[id]
        if pos < self._count - 1:
            self._matrix[pos : self._count - 1] = self._matrix[pos + 1 : self._count]

        del self.ids[pos]
        del self.metadata[id]
        del self._id_to_pos[id]
        self._count -= 1

        # Only entries after the removed row moved; the ones before it did not.
        for i in range(pos, len(self.ids)):
            self._id_to_pos[self.ids[i]] = i

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    def search(self, query: List[float], k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        if self._matrix is None or self._count == 0:
            return []
        q = np.array(query, dtype=np.float32)
        live = self._matrix[: self._count]

        if metric == "cosine":
            norms = np.linalg.norm(live, axis=1, keepdims=True) + 1e-10
            unit_vectors = live / norms
            unit_q = q / (np.linalg.norm(q) + 1e-10)
            scores = unit_vectors @ unit_q
            order = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in order]

        elif metric == "l2":
            dists = np.linalg.norm(live - q, axis=1)
            order = np.argsort(dists)[:k]
            return [(self.ids[i], float(dists[i])) for i in order]

        else:
            raise ValueError(f"unknown metric '{metric}'")
