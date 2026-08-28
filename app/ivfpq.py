"""
IVF + Product Quantization index.

References:
  Jegou, Douze & Schmid, "Product Quantization for Nearest Neighbor Search"
  (IEEE TPAMI, 2011)
  Jegou et al., "Searching in one billion vectors: re-rank with source coding"
  (ICASSP, 2011) -- IVFADC

Why this exists alongside HNSW
------------------------------
HNSW attacks *time*: it avoids scanning all N vectors. But it still stores every
vector at full precision, so memory is O(N * dim * 4 bytes) plus graph overhead.
At 1M x 128-dim that is ~512 MB of raw vectors before any edges.

IVF+PQ attacks *memory*:

  IVF (Inverted File)
      k-means partitions the space into `nlist` Voronoi cells. Each vector is
      assigned to its nearest centroid and stored in that cell's list. A query
      only scans the `nprobe` nearest cells instead of all N vectors.

  PQ (Product Quantization)
      Split each dim-dimensional vector into `m` sub-vectors of length dim/m.
      Each subspace gets its own 256-entry codebook (learned by k-means), so a
      sub-vector is replaced by a single byte. A 128-dim float32 vector
      (512 bytes) becomes m=8 bytes -- 64x compression.

The two approximations are different in kind, which is the point of building
both: HNSW loses recall by not visiting every node; PQ loses recall because the
stored vectors are lossy reconstructions. Their failure modes do not overlap.

Residual encoding
-----------------
PQ encodes `vector - centroid`, not the vector itself. Residuals are smaller and
more tightly distributed than raw vectors, so a fixed 256-entry codebook
represents them far more accurately. This is what makes IVFADC substantially
better than plain PQ.

Asymmetric Distance Computation (ADC)
-------------------------------------
At query time the query is NOT quantized. Instead, for each subspace we
precompute the distance from the query's sub-vector to all 256 codebook entries
-- an (m x 256) lookup table. The distance to any stored vector is then just m
table lookups and a sum: no decompression, no full-precision arithmetic.
Keeping the query exact is where the "asymmetric" comes from, and it is
noticeably more accurate than quantizing both sides.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# k-means (used for both the coarse quantizer and the PQ codebooks)
# ---------------------------------------------------------------------- #


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared L2 distances between every row of X and every row of C.

    Uses ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2 so the whole thing is one
    matrix multiply instead of an (n, k, d) broadcast. At n=100k, k=256, d=128
    the naive version would allocate several GB; this allocates (n, k).
    """
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    c2 = np.einsum("ij,ij->i", C, C)[None, :]
    return np.maximum(x2 - 2.0 * (X @ C.T) + c2, 0.0)


def kmeans(
    X: np.ndarray, k: int, iters: int = 25, seed: int = 0
) -> np.ndarray:
    """Lloyd's algorithm. Returns (k, d) centroids."""
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape

    if n == 0:
        return np.zeros((k, d), dtype=np.float32)

    if n <= k:
        # Fewer points than clusters: use every point, pad by resampling.
        centroids = np.zeros((k, d), dtype=np.float32)
        centroids[:n] = X
        if n < k:
            centroids[n:] = X[rng.integers(0, n, k - n)]
        return centroids

    # k-means++ initialization (Arthur & Vassilvitskii, 2007).
    #
    # Seeding uniformly at random lets two centroids land in the same blob, and
    # Lloyd's iterations cannot recover from it: no centroid can cross the empty
    # space between well-separated clusters, so one ends up stranded midway
    # between two real ones while another cluster is served by two.
    #
    # Sampling each new centroid with probability proportional to D(x)^2 -- the
    # squared distance from x to the NEAREST already-chosen centroid -- biases
    # selection toward whatever region the current set covers worst. That is
    # precisely the failure mode above, so the spread is fixed at seeding time
    # rather than left for Lloyd's to fail to repair.
    centroids = np.zeros((k, d), dtype=np.float32)
    centroids[0] = X[rng.integers(0, n)]

    # Running D(x)^2 against the chosen set, updated by a min() per new centroid
    # so we never recompute distances to centroids already picked.
    closest_sq = _sq_dists(X, centroids[0:1])[:, 0]

    for i in range(1, k):
        weights = closest_sq.astype(np.float64)
        total = weights.sum()
        if total <= 0.0:
            # Every point coincides with an already-chosen centroid (e.g. all
            # rows identical). D^2 carries no signal, so sample uniformly.
            next_idx = int(rng.integers(0, n))
        else:
            next_idx = int(rng.choice(n, p=weights / total))
        centroids[i] = X[next_idx]
        closest_sq = np.minimum(closest_sq, _sq_dists(X, centroids[i : i + 1])[:, 0])

    for _ in range(iters):
        assign = np.argmin(_sq_dists(X, centroids), axis=1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
            else:
                # Empty cluster: reseed on a random point rather than leaving a
                # dead centroid that can never win an assignment again.
                centroids[j] = X[rng.integers(0, n)]

    return centroids


# ---------------------------------------------------------------------- #
# index
# ---------------------------------------------------------------------- #


class IVFPQIndex:
    def __init__(
        self,
        dim: int,
        nlist: int = 64,
        m: int = 8,
        nbits: int = 8,
        metric: str = "cosine",
        seed: int = 0,
    ):
        if dim % m != 0:
            raise ValueError(f"m={m} must divide dim={dim}")
        if nbits != 8:
            raise ValueError("only nbits=8 (256-entry codebooks) is supported")

        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.ksub = 2 ** nbits          # 256 centroids per subspace
        self.dsub = dim // m            # dimensions per subspace
        self.metric = metric
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None    # (nlist, dim)
        self.codebooks: Optional[np.ndarray] = None    # (m, ksub, dsub)
        self.is_trained = False

        # Inverted lists: cell id -> (ids, codes)
        self.lists: Dict[int, List[str]] = {}
        self.codes: Dict[int, List[np.ndarray]] = {}

        self.metadata: Dict[str, dict] = {}
        self.id_to_cell: Dict[str, int] = {}
        self.deleted: set = set()

    # ------------------------------------------------------------------ #
    # preprocessing
    # ------------------------------------------------------------------ #

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        """Normalize for cosine so that L2 becomes a monotone proxy.

        For unit vectors ||a-b||^2 = 2 - 2*cos(a,b), so ranking by L2 is exactly
        ranking by cosine. Everything downstream can then assume L2, which is
        what PQ is defined over.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
            X = X / norms
        return X

    def _to_score(self, sq_dist: float) -> float:
        if self.metric == "cosine":
            # invert ||a-b||^2 = 2 - 2cos  ->  cos = 1 - d^2/2
            return float(1.0 - sq_dist / 2.0)
        return float(np.sqrt(max(sq_dist, 0.0)))

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def train(self, vectors: np.ndarray) -> None:
        """Learn the coarse centroids and the PQ codebooks.

        Unlike HNSW, this index cannot accept a single vector until it has seen
        a representative sample -- the codebooks ARE the compression, and they
        have to be fitted to the data distribution first.
        """
        X = self._prepare(np.asarray(vectors, dtype=np.float32))
        if X.shape[0] < self.nlist:
            raise ValueError(
                f"need at least nlist={self.nlist} training vectors, got {X.shape[0]}"
            )

        # 1. Coarse quantizer: partition the space.
        self.centroids = kmeans(X, self.nlist, seed=self.seed)

        # 2. Residuals: what is left after subtracting the assigned centroid.
        assign = np.argmin(_sq_dists(X, self.centroids), axis=1)
        residuals = X - self.centroids[assign]

        # 3. One codebook per subspace, fitted on that slice of the residuals.
        self.codebooks = np.zeros((self.m, self.ksub, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = residuals[:, j * self.dsub : (j + 1) * self.dsub]
            self.codebooks[j] = kmeans(sub, self.ksub, seed=self.seed + j + 1)

        self.is_trained = True

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #

    def _encode(self, residual: np.ndarray) -> np.ndarray:
        """Residual -> m bytes, one codebook index per subspace."""
        codes = np.zeros(self.m, dtype=np.uint8)
        for j in range(self.m):
            sub = residual[j * self.dsub : (j + 1) * self.dsub].reshape(1, -1)
            codes[j] = np.argmin(_sq_dists(sub, self.codebooks[j]), axis=1)[0]
        return codes

    def add(self, id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        if not self.is_trained:
            raise RuntimeError("index must be trained before adding vectors")
        if id in self.id_to_cell and id not in self.deleted:
            raise ValueError(f"id '{id}' already exists")

        x = self._prepare(np.asarray(vector, dtype=np.float32))[0]
        if x.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {x.shape[0]}")

        cell = int(np.argmin(_sq_dists(x.reshape(1, -1), self.centroids), axis=1)[0])
        residual = x - self.centroids[cell]

        self.lists.setdefault(cell, []).append(id)
        self.codes.setdefault(cell, []).append(self._encode(residual))
        self.id_to_cell[id] = cell
        self.metadata[id] = metadata or {}
        self.deleted.discard(id)

    def delete(self, id: str) -> None:
        if id not in self.id_to_cell:
            raise KeyError(f"id '{id}' not found")
        self.deleted.add(id)

    @property
    def live_count(self) -> int:
        return len(self.id_to_cell) - len(self.deleted)

    # ------------------------------------------------------------------ #
    # search (IVFADC)
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: List[float],
        k: int = 5,
        nprobe: int = 8,
        filter: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        if not self.is_trained:
            raise RuntimeError("index must be trained before searching")
        if not self.id_to_cell:
            return []

        q = self._prepare(np.asarray(query, dtype=np.float32))[0]
        nprobe = max(1, min(nprobe, self.nlist))

        # Which cells to scan: the nprobe nearest coarse centroids.
        coarse = _sq_dists(q.reshape(1, -1), self.centroids)[0]
        probe_cells = np.argsort(coarse)[:nprobe]

        results: List[Tuple[float, str]] = []

        for cell in probe_cells:
            cell = int(cell)
            ids = self.lists.get(cell)
            if not ids:
                continue

            # The residual is relative to THIS cell's centroid, so the lookup
            # table has to be rebuilt per probed cell.
            qr = q - self.centroids[cell]

            # lut[j, t] = ||qr_j - codebook[j][t]||^2
            lut = np.zeros((self.m, self.ksub), dtype=np.float32)
            for j in range(self.m):
                sub = qr[j * self.dsub : (j + 1) * self.dsub].reshape(1, -1)
                lut[j] = _sq_dists(sub, self.codebooks[j])[0]

            # Distance for every vector in the cell = sum of m table lookups.
            # Vectorized: stack codes into (n_cell, m) and fancy-index.
            codes = np.vstack(self.codes[cell])                    # (n_cell, m)
            dists = lut[np.arange(self.m), codes].sum(axis=1)      # (n_cell,)

            for vid, d in zip(ids, dists):
                if vid in self.deleted:
                    continue
                if filter and any(
                    self.metadata.get(vid, {}).get(key) != val
                    for key, val in filter.items()
                ):
                    continue
                results.append((float(d), vid))

        results.sort(key=lambda x: x[0])
        return [(vid, self._to_score(d)) for d, vid in results[:k]]

    # ------------------------------------------------------------------ #
    # memory accounting
    # ------------------------------------------------------------------ #

    def memory_bytes(self) -> Dict[str, int]:
        """Actual storage cost, split into what scales with N and what does not."""
        n = len(self.id_to_cell)
        codes_bytes = n * self.m                       # 1 byte per subspace
        codebook_bytes = self.m * self.ksub * self.dsub * 4
        centroid_bytes = self.nlist * self.dim * 4
        raw_equivalent = n * self.dim * 4              # what float32 would cost

        return {
            "codes": codes_bytes,
            "codebooks": codebook_bytes,
            "coarse_centroids": centroid_bytes,
            "total": codes_bytes + codebook_bytes + centroid_bytes,
            "raw_float32_equivalent": raw_equivalent,
        }

    def compression_ratio(self) -> float:
        """Per-vector compression, excluding the fixed codebook overhead.

        Reported separately because codebooks are a constant cost: negligible at
        1M vectors, dominant at 1k. Folding them into a single headline ratio
        would be misleading at small N.
        """
        return (self.dim * 4) / self.m