from fastapi import FastAPI, HTTPException
from typing import List
from app.models import VectorInsert, SearchQuery, SearchResult
from app.index import FlatIndex

app = FastAPI(title="Mini Vector DB — Week 1: Flat Index")

DIM = 128
index = FlatIndex(dim=DIM)

@app.post("/insert")
def insert_vector(item: VectorInsert):
    try:
        index.insert(item.id, item.vector, item.metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "id": item.id}

@app.post("/search", response_model=List[SearchResult])
def search_vectors(query: SearchQuery):
    results = index.search(query.vector, query.k, query.metric)
    return [
        SearchResult(id=id_, score=score, metadata=index.metadata.get(id_))
        for id_, score in results
    ]

@app.delete("/delete/{id}")
def delete_vector(id: str):
    try:
        index.delete(id)
    except KeyError:
        raise HTTPException(status_code=404, detail="id not found")
    return {"status": "ok", "id": id}

@app.get("/stats")
def stats():
    return {"count": len(index.ids), "dim": index.dim}