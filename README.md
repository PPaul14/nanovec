# nanovec

A vector database built from scratch in Python — flat (exact) search and a
hand-written HNSW graph index, behind a FastAPI service.

## Why

I wanted to understand what FAISS, Milvus and Pinecone are actually doing, so
this implements the internals rather than calling a library. Everything here is
written from the papers: the HNSW layer structure, the Algorithm 4 diversity
heuristic for neighbor selection, and the undirected-edge maintenance that keeps
the graph navigable. A brute-force index runs alongside as the correctness
ground truth, and recall of the approximate index is measured against it rather
than assumed.

The interesting output of a project like this is not the code — it's the
diagnostics. See [Debugging notes](#debugging-notes).

## Current capabilities

- **`FlatIndex`** — brute-force exact search, cosine and L2, NumPy-vectorized.
  Serves as the ground truth for recall measurement.
- **`HNSWIndex`** — multi-layer navigable small world graph built from scratch:
  - skip-list style exponential level assignment
  - greedy descent through sparse upper layers, wide `ef` search at layer 0
  - neighbor selection via the paper's diversity heuristic (Algorithm 4), with
    `keepPrunedConnections` top-up
  - undirected edge maintenance during pruning
- **FastAPI service** — `/insert`, `/search`, `/delete/{id}`, `/stats`.
- **Test suite** — recall validation of HNSW against `FlatIndex` on both uniform
  and clustered data.
- **Diagnostics** — an ef/recall/latency sweep and a graph connectivity checker
  (BFS reachability, component count, edge symmetry, degree distribution,
  cross-cluster edge count).

Known gap: the HTTP API is currently wired only to `FlatIndex`. `HNSWIndex` is
exercised through the tests and benchmarks, not yet through the service.

## Architecture

```mermaid
flowchart TB
    client[Client] -->|HTTP + JSON| api

    subgraph service["FastAPI service (app/main.py)"]
        api["POST /insert · POST /search<br/>DELETE /delete/:id · GET /stats"]
        val["Pydantic schemas<br/>app/models.py"]
        api --> val
    end

    val --> flat

    subgraph indexes["Index layer"]
        flat["FlatIndex — app/index.py<br/>exact · cosine + L2<br/>NumPy vectorized"]
        hnsw["HNSWIndex — app/hnsw.py<br/>approximate · graph traversal"]
    end

    flat -. "ground truth for recall" .-> hnsw
    hnsw -. "not yet exposed via the API" .-> api

    subgraph graph["HNSW layer structure"]
        direction TB
        l2["layer 2 — sparse<br/>long-range hops · entry point"]
        l1["layer 1 — sparse"]
        l0["layer 0 — every node<br/>M0 = 32 · fine-grained"]
        l2 -->|"greedy descent, ef=1"| l1
        l1 -->|"greedy descent, ef=1"| l0
        l0 -->|"wide search, ef budget"| res["top-k"]
    end

    hnsw --> graph
```

Search path through HNSW: enter at the top layer, greedily walk to the closest
node at each level with `ef=1`, then run a wide best-first search at layer 0
with the caller's `ef` budget and return the top `k`.

## Quickstart

```bash
git clone https://github.com/<your-username>/nanovec.git
cd nanovec

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive API docs: <http://127.0.0.1:8000/docs>

Run the tests and diagnostics:

```bash
venv/Scripts/python.exe -m pytest tests/ -v -s
venv/Scripts/python.exe -m benchmarks.ef_curve
venv/Scripts/python.exe -m benchmarks.connectivity
```

## API reference

The service is configured for `DIM = 128`; vectors are abbreviated below.

### `POST /insert`

Request:

```json
{
  "id": "doc_1",
  "vector": [0.12, -0.44, 0.98],
  "metadata": { "source": "wiki", "lang": "en" }
}
```

Response:

```json
{ "status": "ok", "id": "doc_1" }
```

Returns `400` if the dimension does not match or the id already exists.

### `POST /search`

Request — `k` defaults to `5`, `metric` to `"cosine"` (`"l2"` also supported):

```json
{ "vector": [0.10, -0.40, 0.95], "k": 3, "metric": "cosine" }
```

Response:

```json
[
  { "id": "doc_1", "score": 0.9812, "metadata": { "source": "wiki", "lang": "en" } },
  { "id": "doc_7", "score": 0.9433, "metadata": { "source": "wiki", "lang": "en" } },
  { "id": "doc_3", "score": 0.9120, "metadata": null }
]
```

`score` is cosine similarity for `cosine` (higher is closer) and L2 distance for
`l2` (lower is closer).

### `DELETE /delete/{id}`

Response:

```json
{ "status": "ok", "id": "doc_1" }
```

Returns `404` if the id is not present.

### `GET /stats`

Response:

```json
{ "count": 42, "dim": 128 }
```

## Benchmarks

Measured on n=1000, dim=32, cosine, `M=16`, `M0=32`, `ef_construction=200`,
30 queries, Recall@10 against `FlatIndex` as ground truth. Single-threaded
CPython 3.14 on Windows. These are small-scale numbers — see the caveat below.

### Uniform random

| ef | Recall@10 | p50 (ms) | QPS |
|----:|----:|----:|----:|
| 10 | 1.000 | 0.958 | 894.7 |
| 25 | 1.000 | 1.596 | 563.1 |
| 50 | 1.000 | 2.404 | 379.1 |
| 100 | 1.000 | 3.291 | 273.6 |
| 200 | 1.000 | 3.934 | 246.8 |
| 400 | 1.000 | 5.792 | 187.2 |
| **flat (exact)** | **1.000** | **0.102** | **9159.9** |

### Clustered (sigma = 0.05, 10 clusters × 100)

| ef | Recall@10 | p50 (ms) | QPS |
|----:|----:|----:|----:|
| 10 | 1.000 | 0.452 | 2113.2 |
| 25 | 1.000 | 0.499 | 1804.0 |
| 50 | 1.000 | 0.562 | 1711.0 |
| 100 | 1.000 | 0.652 | 1473.1 |
| 200 | 1.000 | 1.282 | 731.0 |
| 400 | 1.000 | 2.481 | 385.1 |
| **flat (exact)** | **1.000** | **0.071** | **12542.0** |

### Graph connectivity (clustered, n=1000)

| Metric | Value |
|---|---|
| Reachable from entry point | 1000 / 1000 |
| Connected components | 1 |
| Layer-0 degree min / mean / max | 8 / 27.9 / 32 |
| Nodes with < 4 edges | 0 |
| Asymmetric edges | 0 |
| Duplicate edges | 0 |
| Cross-cluster edges | 402 / 27874 (1.44%) |
| Layer sizes | L0: 1000 nodes / 27874 edges · L1: 59 / 822 · L2: 3 / 6 |

### The honest caveat

**Brute force currently beats HNSW at this scale, by roughly 5–100×.** That is
the expected result, not a defect. At n=1000 the flat index is one
`(1000, 32) @ (32,)` matrix multiply — a single vectorized C loop over 32k
floats — while HNSW pays Python interpreter overhead per node visited during
graph traversal. The graph's asymptotic advantage (O(log n) nodes visited versus
O(n)) does not pay for that constant factor until the scan itself becomes
expensive.

Recall@10 is 1.000 at every ef on both distributions, so these numbers also say
nothing about the accuracy/speed tradeoff HNSW exists to provide — n=1000 is
simply too small for approximation error to appear. The curves here are
diagnostic, not competitive: what they confirm is that latency now scales with
the ef budget, which is itself a fix (see below).

The crossover is expected in the tens of thousands of vectors. Measuring it at
10k–1M is Week 5 — until then, treat "HNSW is faster" as unproven by this repo.

## Debugging notes

The most useful thing this project produced was a graph diagnostic, and the
reason is that **recall alone did not catch either bug**.

### Act 1 — recall flat at 0.547

HNSW scored a perfect 1.000 Recall@10 on uniform random vectors but only 0.547
on clustered data, and it stayed flat across ef = 10 → 400. Latency barely grew
with ef either.

That combination is diagnostic. If recall were merely *low*, the search would be
too shallow and raising ef would fix it. Recall that is flat **and** latency that
ignores ef means the search is exhausting its reachable set long before it spends
its ef budget — it runs out of graph, not out of budget. That points at
construction, not tuning.

So I wrote `benchmarks/connectivity.py` to test it directly: BFS from the entry
point over layer-0 edges, count components, check edge symmetry. It reported
**only 100 of 1000 nodes reachable** from the entry point, and **37% of layer-0
edges asymmetric**.

Root cause: when a new node A connected to neighbor B, the reverse edge B→A was
added and then immediately pruned away — A was far, so the diversity heuristic
dropped it — leaving a one-way edge A→B. Because the first inserted cluster had
nothing to link outward to, every cross-cluster edge ended up pointing *into* it.
Search could enter a region and never leave, trapped wherever it started.

Fix: maintain undirected edges. When B drops A during pruning, A also drops B.

### Act 2 — 2398 one-way edges survived the fix

After that fix, recall went to 1.000 everywhere and reachability to 1000/1000 —
so by every metric I had originally been watching, the bug was closed. The
connectivity checker disagreed: **2398 of 28776 layer-0 edges were still
asymmetric.**

The undirected-edge repair was correct in intent and defeated by aliasing. The
new node's neighbor list was published by reference and then iterated over
*while the repair mutated it*:

```python
self.neighbors[l][id] = selected   # same list object
for n_id in selected:              # iterating the live list
    ...
    for d in dropped:
        d_list.remove(n_id)        # can remove from `selected` mid-loop
```

When neighbor B pruned the new node A away, the repair called `remove()` on A's
own list — the very list the `for` loop was walking. Removing the current element
shifts the remainder left by one, so **the next neighbor was silently skipped**
and never received its reverse edge. The repair for one-way edges was itself
creating one-way edges.

Fix: iterate a snapshot (`for n_id in list(selected)`). Asymmetric edges went
2398 → 0. Layer-0 edges dropped ~3% (28776 → 27874) as the phantom half-edges
disappeared, and mean degree settled at 27.9.

### What this cost and what it bought

Recall stayed at 1.000 throughout Act 2 — at n=1000 the graph is dense enough
(mean degree ~28, single component) that search succeeds *despite* thousands of
broken edges. A green test suite was actively hiding the defect. The one signal
that would have exposed it earlier is the one in the tables above: latency that
does not respond to ef.

The lesson I'd carry forward: for a data structure, assert on **structural
invariants** — symmetry, reachability, degree bounds — not only on end-to-end
quality metrics. Quality metrics degrade gracefully, which means they hide
structural damage right up until the scale at which they suddenly don't.

## Roadmap

- [x] **Week 1** — flat index (cosine + L2) and FastAPI CRUD service
- [x] **Week 2** — HNSW from scratch, Algorithm 4 heuristic, recall validation
      against the flat baseline, ef sweep and connectivity diagnostics
- [ ] **Week 3** — persistence (snapshot + write-ahead log), metadata filtering,
      and wiring HNSW through the HTTP API
- [ ] **Week 4** — IVF and Product Quantization
- [ ] **Week 5** — benchmarking at 10k–1M vectors; locate the flat/HNSW crossover
- [ ] **Week 6** — Docker, architecture diagrams, documentation

## References

- Malkov, Y. A., & Yashunin, D. A. (2016). *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs.*
  [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest
  Neighbor Search.* IEEE TPAMI, 33(1), 117–128.
  [DOI:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
