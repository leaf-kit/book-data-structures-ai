import numpy as np

from dsai.batching import Request, simulate_continuous, simulate_static, summarize
from dsai.ring import RingBuffer, WindowMean, window_mean_recompute
from dsai.stack import Stack, chain_children, depth_iterative, depth_recursive
from dsai.tape import Tape, forward_mode_grad


def test_stack_lifo():
    s = Stack(4)
    s.push(1); s.push(2); s.push(3)
    assert s.pop() == 3 and s.pop() == 2 and len(s) == 1


def test_depth_agree_small():
    ch = chain_children(50)
    assert depth_recursive(ch) == depth_iterative(ch) == 50


def test_iterative_handles_deep():
    assert depth_iterative(chain_children(100_000)) == 100_000


def test_tape_gradient_matches_numeric():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5)
    w = rng.standard_normal(5)
    t = Tape()
    xi, wi = t.var(x), t.var(w)
    out = t.sum(t.tanh(t.mul(wi, xi)))
    g = t.backward(out)[xi]
    f = lambda v: float(np.tanh(w * v).sum())  # noqa: E731
    for i in range(5):
        e = np.zeros(5); e[i] = 1.0
        assert abs(g[i] - forward_mode_grad(f, x, e)) < 1e-5


def test_tape_is_append_only_and_reversed():
    t = Tape()
    a = t.var(np.array([2.0]))
    b = t.mul(a, a)
    c = t.add(b, a)
    assert [e.out for e in t.entries] == [b, c]
    assert t.backward(c)[a][0] == 5.0   # d(a^2 + a)/da = 2a + 1


def test_ring_buffer_wraps():
    r = RingBuffer(3, dtype=np.int64)
    for v in (1, 2, 3, 4):
        r.push(v)
    assert list(r.view()) == [2, 3, 4]
    assert r.pop() == 2 and list(r.view()) == [3, 4]


def test_window_mean_matches_recompute():
    xs = np.random.default_rng(2).standard_normal(200)
    wm = WindowMean(7)
    got = np.array([wm.push(float(v)) for v in xs])
    assert np.allclose(got, window_mean_recompute(xs, 7))


def test_batching_all_requests_finish():
    rng = np.random.default_rng(3)
    for sim in (simulate_static, simulate_continuous):
        reqs = [Request(float(a), int(t)) for a, t in
                zip(np.cumsum(rng.exponential(5, 50)), rng.integers(1, 20, 50))]
        sim(reqs, 8, 10.0, 0.5)
        assert all(r.done_at >= r.arrival for r in reqs)
        thr, mean, p95 = summarize(reqs)
        assert thr > 0 and p95 >= mean


def test_continuous_not_slower_on_mixed_lengths():
    rng = np.random.default_rng(4)
    arr = np.cumsum(rng.exponential(2, 200))
    tok = rng.integers(1, 100, 200)
    a = [Request(float(x), int(t)) for x, t in zip(arr, tok)]
    b = [Request(float(x), int(t)) for x, t in zip(arr, tok)]
    simulate_static(a, 16, 10.0, 0.5)
    simulate_continuous(b, 16, 10.0, 0.5)
    assert summarize(b)[1] <= summarize(a)[1]
