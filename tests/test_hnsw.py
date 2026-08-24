import random
import numpy as np

from app.index import FlatIndex
from app.hnsw import HNSWIndex


def test_hnsw_recall_against_flat_baseline():
    random.seed(42)
    np.random.seed(42)
    dim, n, k = 32, 500, 10

    flat = FlatIndex(dim=dim)
    hnsw = HNSWIndex(dim=dim, metric="cosine")

    vectors = {}
    for i in range(n):
        vec = np.random.uniform(-1, 1, dim).tolist()
        vid = f"vec_{i}"
        vectors[vid] = vec
        flat.insert(vid, vec)
        hnsw.insert(vid, vec)

    query_ids = random.sample(list(vectors.keys()), 20)
    total_recall = 0.0

    for qid in query_ids:
        q_vec = vectors[qid]
        ground_truth = {id_ for id_, _ in flat.search(q_vec, k=k, metric="cosine")}
        approx = {id_ for id_, _ in hnsw.search(q_vec, k=k)}
        total_recall += len(ground_truth & approx) / k

    avg_recall = total_recall / len(query_ids)
    print(f"\nAverage Recall@{k}: {avg_recall:.3f}")
    assert avg_recall >= 0.85, f"Recall too low: {avg_recall}"


def test_hnsw_recall_on_clustered_data():
    random.seed(7)
    np.random.seed(7)
    dim, k = 32, 10
    n_clusters, per_cluster = 10, 100

    flat = FlatIndex(dim=dim)
    hnsw = HNSWIndex(dim=dim, metric="cosine")

    centers = np.random.uniform(-1, 1, (n_clusters, dim))
    vectors = {}
    for ci, center in enumerate(centers):
        for j in range(per_cluster):
            vec = (center + np.random.normal(0, 0.05, dim)).tolist()
            vid = f"c{ci}_v{j}"
            vectors[vid] = vec
            flat.insert(vid, vec)
            hnsw.insert(vid, vec)

    query_ids = random.sample(list(vectors.keys()), 30)
    total_recall = 0.0
    for qid in query_ids:
        q = vectors[qid]
        truth = {i for i, _ in flat.search(q, k=k, metric="cosine")}
        approx = {i for i, _ in hnsw.search(q, k=k)}
        total_recall += len(truth & approx) / k

    avg_recall = total_recall / len(query_ids)
    print(f"\nClustered Recall@{k}: {avg_recall:.3f}")
    assert avg_recall >= 0.80, f"Recall too low on clustered data: {avg_recall}"