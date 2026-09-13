import numpy as np

from dsai.bst import BST, Treap
from dsai.btree import StaticBTree, expected_height
from dsai.layout import batch_search, binary_search, eytzinger, eytzinger_search
from dsai.learned import PiecewiseLinearIndex


def test_bst_sorted_insert_is_a_chain():
    t = BST(100)
    for k in range(100):
        t.insert(k)
    assert t.height() == 100
    assert t.search(99) == (True, 100)
    assert t.search(1000)[0] is False


def test_treap_stays_shallow():
    t = Treap(5000, seed=1)
    for k in range(5000):
        t.insert(k)
    assert t.height() < 4 * np.log2(5000)
    for k in (0, 17, 4999):
        assert t.search(k)[0]


def test_btree_search_and_height():
    keys = np.arange(0, 100_000, 3, dtype=np.int64)
    t = StaticBTree(keys, 16)
    assert t.height() == expected_height(len(keys), 16)
    for q in (0, 3, 300, 99_999):
        pos, blocks = t.search(q)
        assert keys[pos] <= q and (pos + 1 == len(keys) or keys[pos + 1] > q)
        assert blocks == t.height()


def test_learned_index_error_bound():
    rng = np.random.default_rng(0)
    keys = np.sort(rng.choice(1 << 30, 50_000, replace=False).astype(np.int64))
    idx = PiecewiseLinearIndex(keys, eps=16)
    assert idx.max_error() <= 16
    for k in keys[::997]:
        pos, _ = idx.search(int(k))
        assert keys[pos] == k


def test_layouts_agree():
    keys = np.sort(np.random.default_rng(2).choice(1 << 30, 10_000, replace=False))
    tree = eytzinger(keys)
    kl, tl = keys.tolist(), tree.tolist()
    q = np.random.default_rng(3).choice(keys, 300)
    for k in q:
        assert keys[binary_search(kl, int(k))[0]] == k
        assert tree[eytzinger_search(tl, int(k))[0]] == k
    assert np.array_equal(batch_search(keys, q), np.searchsorted(keys, q))
