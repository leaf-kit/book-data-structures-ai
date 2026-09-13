"""문제 15.8. 블록 안은 닫힌 식, 블록 사이는 재귀. 5장 누적합의 두 단계로 쓴 스캔."""
import numpy as np

from dsai.ssm import run_recurrent, run_scan


def run_scan_blocked(x, a, b, c, block):
    n, s = len(x), len(a)
    idx = np.arange(block)
    diff = idx[:, None] - idx[None, :]
    powtab = a[None, None, :] ** np.clip(diff, 0, None)[:, :, None]
    decay = np.where(diff[:, :, None] >= 0, powtab, 0.0)
    powers = a[None, :] ** np.arange(1, block + 1)[:, None]
    y = np.empty(n)
    h = np.zeros(s)
    for st in range(0, n, block):
        L = min(block, n - st)
        u = b[None, :] * x[st:st + L, None]
        hb = powers[:L] * h[None, :] + np.einsum("ijs,js->is", decay[:L, :L], u)
        y[st:st + L] = hb @ c
        h = hb[-1]
    return y


def test_blocked_scan_matches_recurrence_and_scan():
    rng = np.random.default_rng(3)
    a = np.exp(-rng.uniform(0.01, 0.5, 8))
    b, c = rng.standard_normal(8), rng.standard_normal(8)
    x = rng.standard_normal(1000)
    ref = run_recurrent(x, a, b, c)
    for block in (1, 7, 64, 1000):
        assert np.allclose(run_scan_blocked(x, a, b, c, block), ref, atol=1e-9)
    assert np.allclose(run_scan(x, a, b, c), ref, atol=1e-9)
