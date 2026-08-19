import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self._id_to_pos: Dict[str, int] = {}

    def insert(self, id: str, vector: List[float], metadata: Optional[dict] = None):
        vec = np.array(vector, dtype=np.float32)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
        if id in self._id_to_pos:
            raise ValueError(f"id '{id}' already exists")

        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec])

        self.ids.append(id)
        self._id_to_pos[id] = len(self.ids) - 1
        self.metadata[id] = metadata or {}

    def delete(self, id: str):
        if id not in self._id_to_pos:
            raise KeyError(f"id '{id}' not found")
        pos = self._id_to_pos[id]
        self.vectors = np.delete(self.vectors, pos, axis=0)
        del self.ids[pos]
        del self.metadata[id]
        self._id_to_pos = {id_: i for i, id_ in enumerate(self.ids)}

    def search(self, query: List[float], k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0:
            return []
        q = np.array(query, dtype=np.float32)

        if metric == "cosine":
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
            unit_vectors = self.vectors / norms
            unit_q = q / (np.linalg.norm(q) + 1e-10)
            scores = unit_vectors @ unit_q
            order = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in order]

        elif metric == "l2":
            dists = np.linalg.norm(self.vectors - q, axis=1)
            order = np.argsort(dists)[:k]
            return [(self.ids[i], float(dists[i])) for i in order]

        else:
            raise ValueError(f"unknown metric '{metric}'")
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self._id_to_pos: Dict[str, int] = {}

    def insert(self, id: str, vector: List[float], metadata: Optional[dict] = None):
        vec = np.array(vector, dtype=np.float32)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
        if id in self._id_to_pos:
            raise ValueError(f"id '{id}' already exists")

        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec])

        self.ids.append(id)
        self._id_to_pos[id] = len(self.ids) - 1
        self.metadata[id] = metadata or {}

    def delete(self, id: str):
        if id not in self._id_to_pos:
            raise KeyError(f"id '{id}' not found")
        pos = self._id_to_pos[id]
        self.vectors = np.delete(self.vectors, pos, axis=0)
        del self.ids[pos]
        del self.metadata[id]
        self._id_to_pos = {id_: i for i, id_ in enumerate(self.ids)}

    def search(self, query: List[float], k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0:
            return []
        q = np.array(query, dtype=np.float32)

        if metric == "cosine":
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
            unit_vectors = self.vectors / norms
            unit_q = q / (np.linalg.norm(q) + 1e-10)
            scores = unit_vectors @ unit_q
            order = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in order]

        elif metric == "l2":
            dists = np.linalg.norm(self.vectors - q, axis=1)
            order = np.argsort(dists)[:k]
            return [(self.ids[i], float(dists[i])) for i in order]

        else:
            raise ValueError(f"unknown metric '{metric}'")