"""
HNSW (Hierarchical Navigable Small World) index.

Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs" (2016).

Week 3 additions on top of the Week 2 graph:
  - tombstone deletes (`delete`)  -- nodes stay in the graph for connectivity
    but are excluded from results
  - metadata filtering with two strategies (`filter_mode="pre"` / `"post"`)

Design notes:
  - Multi-layer graph. Upper layers are sparse (long-range hops), layer 0
    contains every node (fine-grained precision).
  - Node levels are drawn from an exponential distribution, like a skip list.
  - Neighbor selection uses the paper's diversity heuristic (Algorithm 4),
    not naive "keep the M nearest", so the graph retains long-range links.
  - Edges are kept UNDIRECTED. When pruning drops A from B's neighbor list,
    B is also dropped from A's list. Without this, pruning silently creates
    one-way edges and search gets trapped in whichever region it started in.
"""

import heapq
import math
import random
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np


_INITIAL_CAPACITY = 1024


class _VectorStore:
    """Dict-like view over the owner's contiguous matrix.

    Exists so `index.data` keeps behaving like the plain dict it used to be --
    `persistence.save_snapshot` iterates it, `load_snapshot` assigns into it,
    and `main._rebuild_flat` reads it. Values are handed back in ORIGINAL
    units: internally cosine vectors are stored unit-length, so the stored row
    is scaled back up by its remembered norm on the way out. Callers therefore
    see exactly what they put in.
    """

    def __init__(self, owner: "HNSWIndex"):
        self._owner = owner

    def __getitem__(self, key: str) -> np.ndarray:
        return self._owner._original_vector(key)

    def __setitem__(self, key: str, value) -> None:
        self._owner._store_vector(key, value)

    def __contains__(self, key: object) -> bool:
        return key in self._owner._row_of

    def __len__(self) -> int:
        return len(self._owner._row_of)

    def __iter__(self) -> Iterator[str]:
        return iter(self._owner._id_at)

    def keys(self) -> List[str]:
        return list(self._owner._id_at)

    def values(self) -> List[np.ndarray]:
        return [self._owner._original_vector(i) for i in self._owner._id_at]

    def items(self) -> List[Tuple[str, np.ndarray]]:
        return [(i, self._owner._original_vector(i)) for i in self._owner._id_at]

    def get(self, key: str, default=None):
        if key in self._owner._row_of:
            return self._owner._original_vector(key)
        return default


class HNSWIndex:
    def __init__(self, dim: int, metric: str = "cosine", M: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1 / math.log(M)

        # One contiguous (capacity, dim) block instead of a dict of thousands of
        # tiny separate arrays. A dict of 10k 64-element arrays costs one Python
        # object header each and scatters them across the heap; a single matrix
        # lets neighbour rows be gathered in one indexing operation and lets the
        # CPU prefetcher do its job. Grown by doubling, so appends amortise O(1).
        self._matrix: np.ndarray = np.zeros((_INITIAL_CAPACITY, dim), dtype=np.float32)
        # Original length of each stored vector. For cosine the matrix holds
        # unit vectors, so this is what reconstructs the caller's input.
        self._norms: np.ndarray = np.ones(_INITIAL_CAPACITY, dtype=np.float32)
        self._row_of: Dict[str, int] = {}
        self._id_at: List[str] = []

        self.data = _VectorStore(self)
        self.metadata: Dict[str, dict] = {}
        self.levels: Dict[str, int] = {}
        self.neighbors: Dict[int, Dict[str, List[str]]] = {}

        # Tombstones. Deleted ids stay in `data` and in the graph so that
        # traversal through them still works -- physically removing a node
        # would tear holes in the connectivity we worked so hard to fix.
        self.deleted: Set[str] = set()

        self.entry_point: Optional[str] = None
        self.max_level: int = -1

    # ------------------------------------------------------------------ #
    # storage
    # ------------------------------------------------------------------ #

    def _ensure_capacity(self, needed: int) -> None:
        capacity = self._matrix.shape[0]
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        used = len(self._id_at)
        grown = np.zeros((capacity, self.dim), dtype=np.float32)
        grown[:used] = self._matrix[:used]
        self._matrix = grown
        grown_norms = np.ones(capacity, dtype=np.float32)
        grown_norms[:used] = self._norms[:used]
        self._norms = grown_norms

    def _store_vector(self, id: str, vector) -> None:
        """Write one vector into the matrix, normalising it for cosine.

        Pre-normalising here is what makes the hot path cheap: with unit rows,
        cosine distance is 1 - dot(a, b), one operation instead of two norm
        computations plus a dot on every single distance evaluation.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")

        if self.metric == "cosine":
            norm = float(np.linalg.norm(vec))
            stored = vec / (norm + 1e-10)
        else:
            norm = 1.0
            stored = vec

        row = self._row_of.get(id)
        if row is None:
            row = len(self._id_at)
            self._ensure_capacity(row + 1)
            self._id_at.append(id)
            self._row_of[id] = row

        self._matrix[row] = stored
        self._norms[row] = norm

    def _original_vector(self, id: str) -> np.ndarray:
        """The vector as the caller supplied it, rebuilt from the unit row."""
        row = self._row_of[id]
        return self._matrix[row] * self._norms[row]

    def _row(self, id: str) -> np.ndarray:
        """The STORED row (unit-length under cosine). Internal use only."""
        return self._matrix[self._row_of[id]]

    def _prepare_query(self, vector) -> np.ndarray:
        """Put an incoming query into the same units as the stored rows."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self.metric == "cosine":
            return vec / (float(np.linalg.norm(vec)) + 1e-10)
        return vec

    # ------------------------------------------------------------------ #
    # distance helpers
    # ------------------------------------------------------------------ #

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Scalar distance between two STORED rows.

        Still needed by the diversity heuristic, which is inherently pairwise.
        No longer called from `_search_layer` -- that batches instead.
        """
        if self.metric == "cosine":
            return 1.0 - float(np.dot(a, b))
        return float(np.linalg.norm(a - b))

    def _distances_to(self, query: np.ndarray, ids: List[str]) -> np.ndarray:
        """Distance from `query` to every id, in ONE vectorised operation.

        The whole point of the contiguous matrix: gather all the neighbour rows
        with a single fancy-index, then one BLAS call. At M0=32 this replaces 32
        separate Python->C round trips with one.
        """
        block = self._matrix[[self._row_of[i] for i in ids]]
        if self.metric == "cosine":
            return 1.0 - (block @ query)
        diff = block - query
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))

    def _to_score(self, dist: float) -> float:
        return 1 - dist if self.metric == "cosine" else dist

    # ------------------------------------------------------------------ #
    # core graph search (used by both insert and search)
    # ------------------------------------------------------------------ #

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[str],
        ef: int,
        layer: int,
        allowed: Optional[Set[str]] = None,
    ) -> List[Tuple[float, str]]:
        """Best-first greedy walk over one layer.

        `candidates` is a min-heap  -> always expand the closest unexplored node.
        `result`     is a max-heap  -> negated distances, so result[0] is the
                                       WORST of the current best ef, which is
                                       what we compare against to decide whether
                                       a newly seen node is worth keeping.

        `allowed` implements IN-GRAPH (pre-)filtering. Crucially, traversal
        still walks through disallowed nodes -- they just never enter the result
        set. Refusing to traverse them would disconnect the graph and reproduce
        exactly the fragmentation bug fixed in Week 2.
        """
        visited = set(entry_points)
        candidates: List[Tuple[float, str]] = []
        result: List[Tuple[float, str]] = []

        def admissible(node: str) -> bool:
            return allowed is None or node in allowed

        if entry_points:
            for ep, d in zip(entry_points, self._distances_to(query, entry_points)):
                d = float(d)
                heapq.heappush(candidates, (d, ep))
                if admissible(ep):
                    heapq.heappush(result, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            # Only stop early once the result set is actually full. Under a
            # selective filter it may stay under ef for a long time, and we
            # must keep exploring rather than bail out.
            if result and len(result) >= ef and dist_c > -result[0][0]:
                break

            # Mark visited while collecting, exactly as the per-neighbour loop
            # did: an id repeated within one adjacency list is still seen once.
            fresh: List[str] = []
            for neighbor in self.neighbors.get(layer, {}).get(c, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    fresh.append(neighbor)
            if not fresh:
                continue

            # One batched distance computation for the whole neighbourhood.
            # Distances do not depend on heap state, so precomputing them and
            # then doing the bookkeeping in order is equivalent to interleaving.
            for neighbor, d in zip(fresh, self._distances_to(query, fresh)):
                d = float(d)

                worth_exploring = (not result) or len(result) < ef or d < -result[0][0]
                if not worth_exploring:
                    continue

                heapq.heappush(candidates, (d, neighbor))
                if admissible(neighbor):
                    heapq.heappush(result, (-d, neighbor))
                    if len(result) > ef:
                        heapq.heappop(result)

        return [(-d, id_) for d, id_ in result]

    # ------------------------------------------------------------------ #
    # neighbor selection (paper Algorithm 4)
    # ------------------------------------------------------------------ #

    def _select_neighbors_heuristic(
        self, base: np.ndarray, candidates: List[Tuple[float, str]], M: int
    ) -> List[str]:
        """Diversity heuristic.

        Skip a candidate if it is closer to an ALREADY-SELECTED neighbor than
        it is to the base node -- that region is already covered, so the edge
        would be redundant. Keeping only the M nearest instead would make every
        edge point into the same dense blob and destroy long-range navigability.
        """
        candidates = sorted(candidates, key=lambda x: x[0])
        selected: List[str] = []

        for dist_to_base, cand_id in candidates:
            if len(selected) >= M:
                break
            cand_vec = self._row(cand_id)
            keep = True
            for sel_id in selected:
                if self._distance(cand_vec, self._row(sel_id)) < dist_to_base:
                    keep = False
                    break
            if keep:
                selected.append(cand_id)

        # keepPrunedConnections: if the heuristic was too aggressive, top back
        # up with the nearest leftovers so the node does not end up starved.
        if len(selected) < M:
            for _, cand_id in candidates:
                if len(selected) >= M:
                    break
                if cand_id not in selected:
                    selected.append(cand_id)

        return selected

    # ------------------------------------------------------------------ #
    # insert
    # ------------------------------------------------------------------ #

    def insert(self, id: str, vector: List[float], metadata: Optional[dict] = None):
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")

        # Re-inserting a tombstoned id resurrects it in place.
        if id in self._row_of and id in self.deleted:
            self.deleted.discard(id)
            self._store_vector(id, vec)
            self.metadata[id] = metadata or {}
            return

        if id in self._row_of:
            raise ValueError(f"id '{id}' already exists")

        self._store_vector(id, vec)
        self.metadata[id] = metadata or {}
        # Everything below compares against stored rows, so switch to the
        # normalised form now rather than mixing units mid-insert.
        vec = self._row(id)

        # Exponentially decaying level, like a skip list: most nodes land on
        # layer 0, a few get promoted high and act as long-range entry points.
        level = int(-math.log(random.random() + 1e-12) * self.mL)
        self.levels[id] = level

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            for l in range(level + 1):
                self.neighbors.setdefault(l, {})[id] = []
            return

        # Phase 1: cheap greedy descent (ef=1) down to the node's own level.
        ep = self.entry_point
        for l in range(self.max_level, level, -1):
            nearest = self._search_layer(vec, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        # Phase 2: real search + connect, from the node's level down to 0.
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vec, [ep], ef=self.ef_construction, layer=l)
            candidates.sort(key=lambda x: x[0])

            max_conn = self.M0 if l == 0 else self.M
            selected = self._select_neighbors_heuristic(vec, candidates, max_conn)
            self.neighbors.setdefault(l, {})[id] = list(selected)

            for n_id in list(selected):
                nb_list = self.neighbors[l].setdefault(n_id, [])
                if id not in nb_list:
                    nb_list.append(id)

                if len(nb_list) > max_conn:
                    n_vec = self._row(n_id)
                    n_candidates = [
                        (self._distance(n_vec, self._row(o)), o) for o in nb_list
                    ]
                    kept = self._select_neighbors_heuristic(n_vec, n_candidates, max_conn)
                    dropped = set(nb_list) - set(kept)
                    self.neighbors[l][n_id] = kept

                    # Keep the graph UNDIRECTED. If B drops A, A must drop B,
                    # otherwise pruning leaves one-way edges and search can get
                    # trapped in the region it started from.
                    for d in dropped:
                        d_list = self.neighbors[l].get(d)
                        if d_list and n_id in d_list:
                            d_list.remove(n_id)

            if candidates:
                ep = candidates[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

    # ------------------------------------------------------------------ #
    # delete (tombstone)
    # ------------------------------------------------------------------ #

    def delete(self, id: str) -> None:
        """Soft delete.

        The node stays in the graph so traversal through it still works; it is
        only filtered out of results. Hard deletion would require repairing
        every neighbour list that pointed at it and risks fragmenting the graph
        -- that is a compaction problem, deferred to a later phase.
        """
        if id not in self._row_of:
            raise KeyError(f"id '{id}' not found")
        self.deleted.add(id)

    @property
    def live_count(self) -> int:
        return len(self._row_of) - len(self.deleted)

    # ------------------------------------------------------------------ #
    # metadata filtering
    # ------------------------------------------------------------------ #

    def _matching_ids(self, filter: Optional[dict]) -> Set[str]:
        """Ids whose metadata matches every key/value in `filter`, minus tombstones.

        This is a full scan -- O(N) per query. Real systems keep an inverted
        index (value -> set of ids) to make this sublinear. Noted as a known
        cost rather than hidden.
        """
        out = set()
        for vid, meta in self.metadata.items():
            if vid in self.deleted:
                continue
            if filter and any(meta.get(k) != v for k, v in filter.items()):
                continue
            out.add(vid)
        return out

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: List[float],
        k: int = 5,
        ef: Optional[int] = None,
        filter: Optional[dict] = None,
        filter_mode: str = "pre",
    ) -> List[Tuple[str, float]]:
        """Approximate k-NN search.

        filter_mode="pre"  -> filter is applied DURING traversal. Slower per
                              query, but reliably returns k results even when
                              the filter is highly selective.
        filter_mode="post" -> search normally, then drop non-matching results.
                              Cheaper, but if few of the true nearest neighbours
                              match the filter you get back fewer than k (or
                              nothing at all). This is the classic ANN filtering
                              trade-off.
        """
        if self.entry_point is None:
            return []
        if filter_mode not in ("pre", "post"):
            raise ValueError("filter_mode must be 'pre' or 'post'")

        ef = ef or max(k, self.ef_construction // 2)
        # Stored rows are unit-length under cosine, so the query has to be put
        # into the same units before any distance is computed against them.
        vec = self._prepare_query(query)

        # Greedy descent through the sparse upper layers. Deliberately
        # unfiltered: descent is only choosing where to start, and constraining
        # it would land us in a worse neighbourhood.
        ep = self.entry_point
        for l in range(self.max_level, 0, -1):
            nearest = self._search_layer(vec, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        if filter_mode == "pre":
            allowed = self._matching_ids(filter)
            if not allowed:
                return []
            candidates = self._search_layer(vec, [ep], ef=ef, layer=0, allowed=allowed)
        else:
            # Oversample so post-filtering has something left to return.
            wide_ef = max(ef, k * 10)
            candidates = self._search_layer(vec, [ep], ef=wide_ef, layer=0)
            allowed = self._matching_ids(filter)
            candidates = [c for c in candidates if c[1] in allowed]

        candidates.sort(key=lambda x: x[0])
        return [(id_, self._to_score(dist)) for dist, id_ in candidates[:k]]