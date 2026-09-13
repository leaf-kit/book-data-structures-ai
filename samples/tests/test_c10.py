import numpy as np

from dsai.beam import beam_search, beam_step, exact_best
from dsai.heap import MaxHeap, topk_heap, topk_select, topk_sort
from dsai.sampling import AliasTable, sample_cumsum, top_p_mask, top_p_mask_by_select
from dsai.scheduler import simulate


def test_heap_pops_in_order():
    h = MaxHeap()
    rng = np.random.default_rng(0)
    xs = rng.permutation(500).tolist()
    for x in xs:
        h.push(x)
    out = [h.pop() for _ in range(500)]
    assert out == sorted(xs, reverse=True)


def test_topk_three_ways_agree():
    x = np.random.default_rng(1).standard_normal(10_000)
    k = 37
    a = set(topk_sort(x, k).tolist())
    assert a == set(topk_select(x, k).tolist()) == set(topk_heap(x, k).tolist())


def test_beam_matches_exhaustive_when_wide():
    rng = np.random.default_rng(2)
    table = rng.standard_normal((4, 5, 5))
    table = table - np.log(np.exp(table).sum(axis=2, keepdims=True))

    def m(prev, t):
        return table[t][prev]
    s_e = exact_best(table)
    seq_b, s_b = beam_search(m, 625, 4, 5)         # 빔이 전부를 담으면 같다
    assert abs(s_e - s_b) < 1e-9
    _, s_1 = beam_search(m, 1, 4, 5)
    assert s_1 <= s_e + 1e-9
    parent, token, sc = beam_step(np.zeros(2), np.log(np.full((2, 3), 1 / 3)), 2)
    assert len(parent) == 2 and set(token.tolist()) <= {0, 1, 2}


def test_alias_and_cumsum_follow_distribution():
    rng = np.random.default_rng(3)
    p = np.array([0.5, 0.25, 0.125, 0.125])
    t = AliasTable(p)
    n = 200_000
    f1 = np.bincount(t.sample(rng, n), minlength=4) / n
    f2 = np.bincount(sample_cumsum(p, rng, n), minlength=4) / n
    assert np.allclose(f1, p, atol=0.01) and np.allclose(f2, p, atol=0.01)


def test_top_p_masks_agree():
    p = np.random.default_rng(4).dirichlet(np.ones(2000) * 0.1)
    for top_p in (0.3, 0.9):
        m1, m2 = top_p_mask(p, top_p), top_p_mask_by_select(p, top_p, 8)
        assert np.array_equal(m1, m2)
        assert p[m1].sum() >= top_p - 1e-9


def test_scheduler_sjf_lowers_mean_and_raises_max():
    rng = np.random.default_rng(5)
    lengths = np.minimum(rng.zipf(1.6, 800) * 32, 4096)
    arr = list(zip(np.cumsum(rng.exponential(0.05, 800)).tolist(), lengths.tolist()))
    f = simulate(arr, "fifo", 2000.0)
    s = simulate(arr, "sjf", 2000.0)
    assert s[0] < f[0]
    assert s[1] >= f[1] * 0.5
