import numpy as np
import pytest

from dsai.freelist import FreeList, SlabPool
from dsai.linked import (IndexList, chain_random, chain_sequential, chase, chase_many,
                         traverse_sum)


def test_chains_are_single_cycles():
    for nxt in (chain_sequential(1000), chain_random(1000, 3)):
        seen, i = set(), 0
        for _ in range(1000):
            assert i not in seen
            seen.add(i)
            i = int(nxt[i])
        assert i == 0 and len(seen) == 1000


def test_chase_and_chase_many_agree():
    nxt = chain_random(500, 4)
    starts = np.array([0, 7, 42])
    many = chase_many(nxt, starts, 33)
    for s, m in zip(starts, many):
        assert chase(nxt, int(s), 33) == m


def test_traverse_sum_equals_array_sum():
    v = np.arange(100, dtype=np.float64)
    nxt = chain_random(100, 5)
    assert traverse_sum(v, nxt, 0, 100) == v.sum()


def test_index_list_insert_and_order():
    lst = IndexList(10)
    a = lst.append(1.0)
    b = lst.append(2.0)
    lst.append(3.0)
    c = lst.insert_after(a, 9.0)
    assert [float(lst.val[i]) for i in lst.order()] == [1.0, 9.0, 2.0, 3.0]
    assert lst.total() == 15.0
    assert c != a and c != b


def test_compact_makes_sequential():
    lst = IndexList(64)
    for i in range(20):
        lst.append(float(i))
    for s in (3, 3, 10):
        lst.insert_after(s, 100.0)
    before = [float(lst.val[i]) for i in lst.order()]
    lst.compact()
    after = [float(lst.val[i]) for i in lst.order()]
    assert before == after
    assert np.array_equal(lst.order(), np.arange(23))
    assert lst.total() == sum(before)


def test_free_list_lifo():
    fl = FreeList(4)
    a, b = fl.alloc(), fl.alloc()
    assert (a, b) == (0, 1)
    fl.free(a)
    assert fl.alloc() == a          # 방금 놓은 칸이 먼저 나온다
    fl.alloc(); fl.alloc()
    with pytest.raises(MemoryError):
        fl.alloc()


def test_slab_pool_classes():
    pool = SlabPool([16, 64, 256], 4)
    assert pool.size_class(17) == 64
    s, i = pool.alloc(100)
    assert s == 256
    pool.free(s, i)
    waste = pool.internal_waste(np.array([16, 17, 65]))
    assert 0 < waste < 1
