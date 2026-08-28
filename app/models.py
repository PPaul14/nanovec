"""Pydantic request/response schemas."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class VectorInsert(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None


class SearchQuery(BaseModel):
    vector: List[float]
    k: int = Field(default=5, ge=1)
    metric: str = "cosine"

    # Which index answers the query. Keeping both live lets you compare the
    # approximate result against exact ground truth over the same data.
    backend: Literal["hnsw", "flat"] = "hnsw"

    # HNSW search breadth. None -> index default.
    ef: Optional[int] = Field(default=None, ge=1)

    # Exact-match metadata filter, e.g. {"category": "shoes", "in_stock": True}
    filter: Optional[Dict[str, Any]] = None

    # "pre"  -> filter during graph traversal: reliably returns k results,
    #           costs more per query
    # "post" -> filter after search: cheaper, but may return fewer than k
    filter_mode: Literal["pre", "post"] = "pre"


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    live_vectors: int
    tombstoned: int
    dim: int
    max_level: int
    entry_point: Optional[str]
    wal_bytes: int


class OkResponse(BaseModel):
    status: str
    id: Optional[str] = None
    detail: Optional[str] = None