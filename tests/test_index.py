from app.index import FlatIndex

def test_insert_and_search_cosine():
    idx = FlatIndex(dim=3)
    idx.insert("a", [1, 0, 0])
    idx.insert("b", [0, 1, 0])
    idx.insert("c", [0.9, 0.1, 0])

    results = idx.search([1, 0, 0], k=2, metric="cosine")
    ids = [r[0] for r in results]

    assert ids[0] == "a"     # exact match should rank first
    assert "c" in ids        # near-match should also show up

def test_delete_removes_vector():
    idx = FlatIndex(dim=2)
    idx.insert("x", [1, 1])
    idx.delete("x")
    assert idx.search([1, 1], k=1) == []

def test_duplicate_id_rejected():
    idx = FlatIndex(dim=2)
    idx.insert("x", [1, 1])
    try:
        idx.insert("x", [2, 2])
        assert False, "should have raised"
    except ValueError:
        pass