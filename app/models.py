from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class VectorInsert(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    vector: List[float]
    k: int = Field(default=5, ge=1)
    metric: str = "cosine"

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None