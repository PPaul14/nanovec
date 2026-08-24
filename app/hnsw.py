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

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            return 1 - float(np.dot(a, b) / denom)
        return float(np.linalg.norm(a - b))

    def _to_score(self, dist: float) -> float:
        return 1 - dist if self.metric == "cosine" else dist

    def _search_layer(self, query: np.ndarray, entry_points: List[str], ef: int, layer: int) -> List[Tuple[float, str]]:
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

    def insert(self, id: str, vector: List[float], metadata: Optional[dict] = None):
        vec = np.array(vector, dtype=np.float32)
        self.data[id] = vec
        self.metadata[id] = metadata or {}

        level = int(-math.log(random.random() + 1e-12) * self.mL)
        self.levels[id] = level

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            for l in range(level + 1):
                self.neighbors.setdefault(l, {})[id] = []
            return

        ep = self.entry_point
        for l in range(self.max_level, level, -1):
            nearest = self._search_layer(vec, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vec, [ep], ef=self.ef_construction, layer=l)
            candidates.sort(key=lambda x: x[0])

            max_conn = self.M0 if l == 0 else self.M
            selected = [c_id for _, c_id in candidates[:max_conn]]

            self.neighbors.setdefault(l, {})[id] = selected
            for n_id in selected:
                self.neighbors[l].setdefault(n_id, []).append(id)
                if len(self.neighbors[l][n_id]) > max_conn:
                    n_vec = self.data[n_id]
                    ranked = sorted(
                        self.neighbors[l][n_id],
                        key=lambda other: self._distance(n_vec, self.data[other]),
                    )
                    self.neighbors[l][n_id] = ranked[:max_conn]

            if candidates:
                ep = candidates[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

    def search(self, query: List[float], k: int = 5, ef: Optional[int] = None) -> List[Tuple[str, float]]:
        if self.entry_point is None:
            return []
        ef = ef or max(k, self.ef_construction // 2)
        vec = np.array(query, dtype=np.float32)

        ep = self.entry_point
        for l in range(self.max_level, 0, -1):
            nearest = self._search_layer(vec, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        candidates = self._search_layer(vec, [ep], ef=ef, layer=0)
        candidates.sort(key=lambda x: x[0])

        return [(id_, self._to_score(dist)) for dist, id_ in candidates[:k]]