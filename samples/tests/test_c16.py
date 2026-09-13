import hashlib

import numpy as np

from dsai.collective import allreduce_ring, allreduce_star, bytes_per_machine
from dsai.loader import (block_shuffle, feistel_permutation, fisher_yates,
                         sequential_reads)
from dsai.merkle import ContentStore, block_hashes, changed_blocks, merkle_root


def test_ring_equals_star_and_moves_less():
    rng = np.random.default_rng(0)
    for p in (2, 3, 8):
        shards = [rng.standard_normal(101) for _ in range(p)]
        t1, b1, _ = allreduce_star(shards)
        t2, b2, steps = allreduce_ring(shards)
        assert np.allclose(t1, sum(shards)) and np.allclose(t2, sum(shards))
        assert b1 == bytes_per_machine(101, 8, p, "star")
        assert steps == 2 * (p - 1)
        if p > 2:
            assert b2 < b1


def test_merkle_finds_changed_blocks():
    rng = np.random.default_rng(1)
    d = bytes(rng.integers(0, 256, 10000, dtype=np.uint8))
    d2 = bytearray(d)
    d2[3000] ^= 1
    d2[8100] ^= 1
    la = merkle_root(block_hashes(d, 256))[1]
    lb = merkle_root(block_hashes(bytes(d2), 256))[1]
    assert changed_blocks(la, lb) == [3000 // 256, 8100 // 256]
    assert changed_blocks(la, la) == []
    assert la[0][0] == hashlib.sha256(d[:256]).digest()
    store = ContentStore()
    h1 = store.put(d, 256)
    h2 = store.put(bytes(d2), 256)
    assert store.stored == len(d) + 2 * 256
    assert store.get(h1) == d and store.get(h2) == bytes(d2)


def test_permutations_are_bijections():
    rng = np.random.default_rng(2)
    assert np.array_equal(np.sort(fisher_yates(300, rng)), np.arange(300))
    bs = block_shuffle(1000, 100, rng)
    assert np.array_equal(np.sort(bs), np.arange(1000))
    assert sequential_reads(bs, 100) > sequential_reads(rng.permutation(1000), 100)
    keys = rng.integers(0, 2**31, 4)
    for n in (1, 2, 5, 1000, 70000):
        p = feistel_permutation(np.arange(n), n, keys)
        assert np.array_equal(np.sort(p), np.arange(n))
    a = feistel_permutation(np.array([7]), 1000, keys)
    assert a[0] == feistel_permutation(np.arange(1000), 1000, keys)[7]
