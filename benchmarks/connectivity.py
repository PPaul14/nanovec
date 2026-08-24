"""
Diagnostic: is the HNSW layer-0 graph connected?

Run from the project root (vector-db/):
    venv/Scripts/python.exe -m benchmarks.connectivity

Hypothesis being tested:
    Clustered recall is flat across ef, and latency does not grow with ef.
    That means search exhausts the reachable node set before spending its ef
    budget -> the graph has disconnected components.

What it reports:
    - reachable node count from the entry point (BFS over layer-0 edges)
    - number of separate connected components
    - degree statistics (are nodes getting their M connections at all?)
    - edge symmetry (HNSW edges should be bidirectional)
    - cross-cluster edge count (the long-range links that make search work)
"""

from collections import deque

import numpy as np

from app.hnsw import HNSWIndex


def build_clustered(dim=32, n_clusters=10, per_cluster=100, sigma=0.05, seed=7):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, (n_clusters, dim))
    vectors = {}
    for ci, center in enumerate(centers):
        for j in range(per_cluster):
            vectors[f"c{ci}_v{j}"] = (center + rng.normal(0, sigma, dim)).tolist()
    return vectors


def bfs_reachable(hnsw, start, layer=0):
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in hnsw.neighbors.get(layer, {}).get(node, []):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return seen


def count_components(hnsw, all_ids, layer=0):
    unvisited = set(all_ids)
    sizes = []
    while unvisited:
        seed_node = next(iter(unvisited))
        comp = bfs_reachable(hnsw, seed_node, layer)
        comp &= set(all_ids)
        sizes.append(len(comp))
        unvisited -= comp
    return sorted(sizes, reverse=True)


def cluster_of(vid):
    return vid.split("_")[0]


def main():
    vectors = build_clustered()
    hnsw = HNSWIndex(dim=32, metric="cosine")
    for vid, vec in vectors.items():
        hnsw.insert(vid, vec)

    all_ids = list(vectors.keys())
    n = len(all_ids)
    layer0 = hnsw.neighbors.get(0, {})

    print(f"\n=== HNSW layer-0 connectivity ({n} nodes) ===")

    reachable = bfs_reachable(hnsw, hnsw.entry_point, layer=0)
    print(f"entry point            : {hnsw.entry_point}")
    print(f"reachable from entry   : {len(reachable)} / {n}")
    if len(reachable) < n:
        print(f"  -> UNREACHABLE       : {n - len(reachable)} nodes "
              f"({100 * (n - len(reachable)) / n:.1f}%)  <-- FRAGMENTED")
    else:
        print("  -> graph is fully connected from the entry point")

    comps = count_components(hnsw, all_ids, layer=0)
    print(f"connected components   : {len(comps)}")
    print(f"component sizes (top 10): {comps[:10]}")

    degrees = [len(layer0.get(v, [])) for v in all_ids]
    print("\n=== degree stats (layer 0, M0 = %d) ===" % hnsw.M0)
    print(f"min / mean / max       : {min(degrees)} / {sum(degrees)/len(degrees):.1f} / {max(degrees)}")
    print(f"nodes with 0 edges     : {sum(1 for d in degrees if d == 0)}")
    print(f"nodes with < 4 edges   : {sum(1 for d in degrees if d < 4)}")

    asym = 0
    dupes = 0
    for v in all_ids:
        nbs = layer0.get(v, [])
        if len(nbs) != len(set(nbs)):
            dupes += 1
        for nb in nbs:
            if v not in layer0.get(nb, []):
                asym += 1
    print(f"\nasymmetric edges       : {asym}   (should be 0 - HNSW edges are bidirectional)")
    print(f"nodes with dup edges   : {dupes}   (should be 0)")

    cross = 0
    total_edges = 0
    for v in all_ids:
        for nb in layer0.get(v, []):
            total_edges += 1
            if cluster_of(v) != cluster_of(nb):
                cross += 1
    print(f"\ncross-cluster edges    : {cross} / {total_edges} "
          f"({100 * cross / max(total_edges, 1):.2f}%)")
    print("  -> these are the long-range links that let search hop between clusters.")
    print("     Near zero here explains why search cannot leave its starting cluster.\n")

    print("=== upper layers ===")
    for l in sorted(hnsw.neighbors.keys(), reverse=True):
        nodes_at_l = len(hnsw.neighbors[l])
        edges_at_l = sum(len(x) for x in hnsw.neighbors[l].values())
        print(f"  layer {l:>2}: {nodes_at_l:>5} nodes, {edges_at_l:>6} edges")
    print(f"  max_level = {hnsw.max_level}\n")


if __name__ == "__main__":
    main()