import numpy as np

from dsai import roofline as rl
from dsai.cost import (blocks_touched, gather_sum, moves_random_expected,
                       moves_sequential, random_index, sequential_index)
from dsai.knn import knn_bruteforce, random_corpus, recall_at_k


def test_gather_sum_same_answer_any_order():
    v = np.arange(1000, dtype=np.float32)
    assert gather_sum(v, sequential_index(1000)) == gather_sum(v, random_index(1000, 1))


def test_random_index_is_permutation():
    idx = random_index(500, 3)
    assert sorted(idx.tolist()) == list(range(500))


def test_blocks_sequential_matches_bound():
    idx = sequential_index(1024)
    assert blocks_touched(idx, 4, 64) == moves_sequential(1024, 4, 64) == 64


def test_blocks_random_is_larger():
    n = 100_000
    seq = blocks_touched(sequential_index(n), 4, 64)
    rnd = blocks_touched(random_index(n, 7), 4, 64)
    assert rnd > 10 * seq


def test_moves_random_expected_bounds():
    lines = 8 * 1024 * 1024 // 64
    e = moves_random_expected(10_000_000, 4, 64, cache_lines=lines)
    assert moves_sequential(10_000_000, 4, 64) < e <= 10_000_000 + lines
    assert moves_random_expected(1000, 4, 64, cache_lines=lines) == 63


def test_intensity_ordering():
    n = 1024
    i_dot = rl.intensity(rl.flops_dot(n), rl.bytes_dot(n, 4))
    i_mv = rl.intensity(rl.flops_matvec(n, n), rl.bytes_matvec(n, n, 4))
    i_mm = rl.intensity(rl.flops_matmul(n, n, n), rl.bytes_matmul(n, n, n, 4))
    assert i_dot == 0.25
    assert i_dot < i_mv < i_mm


def test_attainable_is_min_of_two_roofs():
    assert rl.attainable(100.0, 10.0, 1.0) == 10.0
    assert rl.attainable(100.0, 10.0, 100.0) == 100.0
    assert rl.ridge_point(100.0, 10.0) == 10.0


def test_knn_exact_recall_one():
    xs = random_corpus(2000, 32, 5)
    q = xs[17]
    found = knn_bruteforce(xs, q, 5)
    assert found[0] == 17
    assert recall_at_k(found, found) == 1.0


def test_knn_sorted_descending():
    xs = random_corpus(500, 16, 9)
    q = random_corpus(1, 16, 10)[0]
    found = knn_bruteforce(xs, q, 7)
    s = xs[found] @ q
    assert np.all(np.diff(s) <= 1e-6)
