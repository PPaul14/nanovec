"""
HNSW (Hierarchical Navigable Small World) index.

Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs" (2016).

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
from typing import Dict, List, Optional, Tuple

import numpy as np


class HNSWIndex:
    def __init__(self, dim: int, metric: str = "cosine", M: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1 / math.log(M)

        self.data: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}
        self.levels: Dict[str, int] = {}
        self.neighbors: Dict[int, Dict[str, List[str]]] = {}

        self.entry_point: Optional[str] = None
        self.max_level: int = -1

    # ------------------------------------------------------------------ #
    # distance helpers
    # ------------------------------------------------------------------ #

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            return 1 - float(np.dot(a, b) / denom)
        return float(np.linalg.norm(a - b))

    def _to_score(self, dist: float) -> float:
        return 1 - dist if self.metric == "cosine" else dist

    # ------------------------------------------------------------------ #
    # core graph search (used by both insert and search)
    # ------------------------------------------------------------------ #

    def _search_layer(
        self, query: np.ndarray, entry_points: List[str], ef: int, layer: int
    ) -> List[Tuple[float, str]]:
        """Best-first greedy walk over one layer.

        `candidates` is a min-heap  -> always expand the closest unexplored node.
        `result`     is a max-heap  -> negated distances, so result[0] is the
                                       WORST of the current best ef, which is
                                       what we compare against to decide whether
                                       a newly seen node is worth keeping.
        """
        visited = set(entry_points)
        candidates: List[Tuple[float, str]] = []
        result: List[Tuple[float, str]] = []

        for ep in entry_points:
            d = self._distance(query, self.data[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(result, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            furthest_dist = -result[0][0]
            # Nothing left that could improve the result set -> stop early.
            if dist_c > furthest_dist and len(result) >= ef:
                break

            for neighbor in self.neighbors.get(layer, {}).get(c, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._distance(query, self.data[neighbor])
                furthest_dist = -result[0][0]
                if len(result) < ef or d < furthest_dist:
                    heapq.heappush(candidates, (d, neighbor))
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
            cand_vec = self.data[cand_id]
            keep = True
            for sel_id in selected:
                if self._distance(cand_vec, self.data[sel_id]) < dist_to_base:
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
        vec = np.array(vector, dtype=np.float32)
        self.data[id] = vec
        self.metadata[id] = metadata or {}

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
            self.neighbors.setdefault(l, {})[id] = selected

            for n_id in selected:
                nb_list = self.neighbors[l].setdefault(n_id, [])
                if id not in nb_list:
                    nb_list.append(id)

                if len(nb_list) > max_conn:
                    n_vec = self.data[n_id]
                    n_candidates = [
                        (self._distance(n_vec, self.data[o]), o) for o in nb_list
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
    # search
    # ------------------------------------------------------------------ #

    def search(
        self, query: List[float], k: int = 5, ef: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        if self.entry_point is None:
            return []
        ef = ef or max(k, self.ef_construction // 2)
        vec = np.array(query, dtype=np.float32)

        # Greedy descent through the sparse upper layers...
        ep = self.entry_point
        for l in range(self.max_level, 0, -1):
            nearest = self._search_layer(vec, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        # ...then a wide, precise search at layer 0.
        candidates = self._search_layer(vec, [ep], ef=ef, layer=0)
        candidates.sort(key=lambda x: x[0])

        return [(id_, self._to_score(dist)) for dist, id_ in candidates[:k]]
