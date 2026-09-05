"""
FastAPI service.

Week 3 changes:
  - HNSW is now the default query backend (previously HNSW existed only in
    tests, so the interesting half of the project was unreachable via the API).
  - FlatIndex is still maintained in parallel and reachable via
    `"backend": "flat"`, so exact ground truth can be compared against the
    approximate result over identical data at runtime.
  - Writes go through a WAL and the index is recovered from disk on startup.

Concurrency caveat, stated rather than hidden: the index is plain Python
objects with no locking, and FastAPI can interleave requests. Single worker
(the default) is safe. Multi-worker would need either a lock or a single
writer process. Proper concurrency control is a later phase.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException

from app.hnsw import HNSWIndex
from app.index import FlatIndex
from app.models import (
    OkResponse,
    SearchQuery,
    SearchResult,
    StatsResponse,
    VectorInsert,
)
from app.persistence import WAL_FILE, WriteAheadLog, recover, save_snapshot

DIM = 128
DATA_DIR = Path("data")

state: Dict[str, object] = {}


def _rebuild_flat(hnsw: HNSWIndex) -> FlatIndex:
    """Mirror the recovered HNSW contents into a fresh FlatIndex.

    Only the HNSW index is snapshotted; the flat index is derived state and is
    cheap to rebuild, so persisting it too would just be duplicated bytes.
    """
    flat = FlatIndex(dim=hnsw.dim)
    for vid, vec in hnsw.data.items():
        if vid not in hnsw.deleted:
            flat.insert(vid, vec.tolist(), hnsw.metadata.get(vid))
    return flat


@asynccontextmanager
async def lifespan(app: FastAPI):
    hnsw = recover(DATA_DIR, dim=DIM)
    state["hnsw"] = hnsw
    state["flat"] = _rebuild_flat(hnsw)
    state["wal"] = WriteAheadLog(DATA_DIR / WAL_FILE)
    yield
    # Clean shutdown: snapshot so the next boot does not replay the whole log.
    save_snapshot(state["hnsw"], DATA_DIR)
    state["wal"].truncate()


app = FastAPI(title="nanovec", version="1.0.0", lifespan=lifespan)


@app.post("/insert", response_model=OkResponse)
def insert_vector(item: VectorInsert):
    hnsw: HNSWIndex = state["hnsw"]
    flat: FlatIndex = state["flat"]
    wal: WriteAheadLog = state["wal"]

    if len(item.vector) != DIM:
        raise HTTPException(400, f"expected dim {DIM}, got {len(item.vector)}")
    if item.id in hnsw.data and item.id not in hnsw.deleted:
        raise HTTPException(409, f"id '{item.id}' already exists")

    # WRITE-AHEAD: log first, apply second. If the process dies between the two,
    # recovery replays the record and the write survives. Applying first would
    # mean an acknowledged write could vanish.
    wal.append(
        {
            "op": "insert",
            "id": item.id,
            "vector": item.vector,
            "metadata": item.metadata,
        }
    )

    try:
        hnsw.insert(item.id, item.vector, item.metadata)
        flat.insert(item.id, item.vector, item.metadata)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return OkResponse(status="ok", id=item.id)


@app.post("/search", response_model=List[SearchResult])
def search_vectors(query: SearchQuery):
    hnsw: HNSWIndex = state["hnsw"]
    flat: FlatIndex = state["flat"]

    if len(query.vector) != DIM:
        raise HTTPException(400, f"expected dim {DIM}, got {len(query.vector)}")

    if query.backend == "hnsw":
        results = hnsw.search(
            query.vector,
            k=query.k,
            ef=query.ef,
            filter=query.filter,
            filter_mode=query.filter_mode,
        )
        meta_source = hnsw.metadata
    else:
        # FlatIndex has no filtering of its own, so filter its exact results
        # afterwards. Correct by construction -- it scanned everything anyway.
        raw = flat.search(query.vector, k=query.k * 10, metric=query.metric)
        allowed = hnsw._matching_ids(query.filter)
        results = [(i, s) for i, s in raw if i in allowed][: query.k]
        meta_source = flat.metadata

    return [
        SearchResult(id=i, score=s, metadata=meta_source.get(i)) for i, s in results
    ]


@app.delete("/delete/{id}", response_model=OkResponse)
def delete_vector(id: str):
    hnsw: HNSWIndex = state["hnsw"]
    flat: FlatIndex = state["flat"]
    wal: WriteAheadLog = state["wal"]

    if id not in hnsw.data or id in hnsw.deleted:
        raise HTTPException(404, "id not found")

    wal.append({"op": "delete", "id": id})

    hnsw.delete(id)  # tombstone; node stays in the graph for connectivity
    try:
        flat.delete(id)  # flat can remove outright, it has no graph to preserve
    except KeyError:
        pass

    return OkResponse(status="ok", id=id)


@app.post("/snapshot", response_model=OkResponse)
def snapshot():
    """Persist current state and truncate the log.

    Order matters: snapshot must be durable BEFORE the WAL is dropped, or a
    crash in between loses every mutation the log was holding.
    """
    hnsw: HNSWIndex = state["hnsw"]
    wal: WriteAheadLog = state["wal"]

    save_snapshot(hnsw, DATA_DIR)
    wal.truncate()
    return OkResponse(status="ok", detail=f"snapshot written to {DATA_DIR}")


@app.get("/stats", response_model=StatsResponse)
def stats():
    hnsw: HNSWIndex = state["hnsw"]
    wal: WriteAheadLog = state["wal"]
    return StatsResponse(
        live_vectors=hnsw.live_count,
        tombstoned=len(hnsw.deleted),
        dim=hnsw.dim,
        max_level=hnsw.max_level,
        entry_point=hnsw.entry_point,
        wal_bytes=wal.size(),
    )