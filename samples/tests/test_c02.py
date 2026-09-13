import numpy as np

from dsai.alloc import ContiguousArena, PagedArena
from dsai.dynarray import DynArray, total_copied
from dsai.kvcache import BlockPool, ContiguousKV, PagedKV, bytes_per_token


def test_dynarray_doubling_amortized():
    a = DynArray(factor=2.0, capacity=4)
    for i in range(1000):
        a.append(i)
    assert a.size == 1000 and a.capacity == 1024
    # 옮긴 원소 수는 2n 을 넘지 않는다 (정리 2.1)
    assert a.copied_bytes // 4 <= 2 * 1000
    assert np.array_equal(a.view(), np.arange(1000, dtype=np.float32))


def test_total_copied_bound():
    assert total_copied(1000, 2.0, 4) == 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512
    assert total_copied(10**6, 2.0, 4) < 2 * 10**6
    assert total_copied(10**6, 1.125, 4) > 5 * 10**6


def test_fixed_increment_is_quadratic():
    small = DynArray(increment=100, capacity=100)
    for i in range(1000):
        small.append(i)
    assert small.copied_bytes // 4 == sum(range(100, 1000, 100))


def test_contiguous_arena_fragments():
    a = ContiguousArena(100)
    assert a.alloc(1, 30) and a.alloc(2, 30) and a.alloc(3, 30)
    a.release(2)
    assert a.free_total() == 40 and a.largest_free() == 30
    assert not a.alloc(4, 40)          # 총합은 40 인데 40 을 못 준다
    assert a.external_fragmentation() > 0
    a.release(3)
    assert a.largest_free() == 70      # 이웃이 합쳐진다


def test_paged_arena_no_external():
    p = PagedArena(100, 10)
    assert p.alloc(1, 25) and p.alloc(2, 25) and p.alloc(3, 25)
    p.release(2)
    assert p.alloc(4, 40)              # 페이지가 흩어져 있어도 준다
    assert 0 < p.internal_fragmentation() < 0.2


def test_bytes_per_token():
    assert bytes_per_token(40, 40, 128, 2) == 819_200


def test_paged_kv_gather_matches_contiguous():
    pool = BlockPool(num_blocks=8, block_size=4, dim=3)
    c = ContiguousKV(max_len=20, dim=3)
    p = PagedKV(pool)
    rng = np.random.default_rng(0)
    for _ in range(10):
        kv = rng.standard_normal(3).astype(np.float32)
        c.append(kv)
        p.append(kv)
    assert np.array_equal(c.gather(), p.gather())
    assert p.reserved() == 12 and c.reserved() == 20


def test_fork_shares_and_copies_on_write():
    pool = BlockPool(num_blocks=16, block_size=4, dim=2)
    root = PagedKV(pool)
    for i in range(6):
        root.append(np.full(2, i, dtype=np.float32))
    child = root.fork()
    assert pool.used_blocks() == 2
    child.append(np.full(2, 99, dtype=np.float32))
    # 마지막 블록만 복사됐다. 첫 블록은 여전히 공유
    assert pool.used_blocks() == 3
    assert root.gather()[5, 0] == 5 and child.gather()[6, 0] == 99
    assert root.length == 6 and child.length == 7
    child.release()
    root.release()
    assert pool.used_blocks() == 0
