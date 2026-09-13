import numpy as np

from dsai.diskgraph import PagedGraph, search_paged, search_with_pq_filter
from dsai.kdtree import KDTree, brute_nearest
from dsai.nsw import HNSW, NSW, clustered_points
from dsai.skiplist import SkipList


def test_kdtree_exact_in_low_dim():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((5000, 3)).astype(np.float32)
    tree = KDTree(pts, leaf_size=8)
    for q in rng.standard_normal((50, 3)).astype(np.float32):
        i, seen = tree.nearest(q)
        assert i == brute_nearest(pts, q)
        assert seen < 5000


def test_skiplist_search_and_order():
    sl = SkipList(2000, p=0.5, seed=3)
    keys = np.random.default_rng(1).permutation(2000)
    for k in keys:
        sl.insert(int(k))
    assert all(sl.search(int(k))[0] for k in keys[:200])
    assert not sl.search(5000)[0]
    cur, prev = sl.nxt[0, sl.head], -1
    while cur != -1:
        assert sl.key[cur] > prev
        prev, cur = sl.key[cur], sl.nxt[0, cur]


def test_graph_search_finds_neighbors():
    rng = np.random.default_rng(2)
    pts = clustered_points(3000, 16, 20, rng)
    order = rng.permutation(3000)
    nsw = NSW(pts, 8).build(order)
    q = pts[7] + 0.001
    res, nd = nsw.search(q, 32)
    assert 7 in res[:3]
    assert nd < 3000
    assert max(len(l) for l in nsw.nbrs) <= 16
    h = HNSW(pts, 8, rng).build(order)
    res2, _ = h.search(q, 32)
    assert 7 in res2[:3]


def test_paged_reads_fewer_when_colocated():
    rng = np.random.default_rng(4)
    pts = clustered_points(2000, 8, 10, rng)
    nbrs = NSW(pts, 10).build(rng.permutation(2000)).nbrs
    q = pts[11] + 0.001
    a = search_paged(PagedGraph(pts, nbrs, colocated=False), q, 0, 32)
    b = search_paged(PagedGraph(pts, nbrs, colocated=True), q, 0, 32)
    assert a[0] == b[0]                                   # 배치는 답을 안 바꾼다
    assert b[1] < a[1]
    coarse = np.round(pts * 4) / 4
    c = search_with_pq_filter(PagedGraph(pts, nbrs, colocated=True), coarse, q, 0, 32)
    assert c[1] <= b[1]
