"""문제 15.7. 지수 감쇠 누적 점수. 옛 참조를 잊는 헤비 히터."""
import numpy as np

from dsai.kvevict import decode_with_policy


def decode_with_decay(q, k, v, budget, lam, recent=0):
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    keep = np.zeros(n, dtype=bool)
    score = np.zeros(n)
    out = np.zeros((n, v.shape[1]))
    for t in range(n):
        keep[t] = True
        idx = np.flatnonzero(keep)
        s = (k[idx] @ q[t]) * scale
        p = np.exp(s - s.max())
        p /= p.sum()
        out[t] = p @ v[idx]
        score[idx] = lam * score[idx] + p                  # lam = 1 이면 누적 합 그대로
        if len(idx) > budget:
            cand = idx[: max(len(idx) - recent, 1)]
            keep[cand[np.argmin(score[cand])]] = False
    return out


def test_lambda_one_equals_heavy_hitter():
    rng = np.random.default_rng(2)
    q, k, v = (rng.standard_normal((80, 16)) for _ in range(3))
    ref, _ = decode_with_policy(q, k, v, 20, "heavy", recent=4)
    assert np.allclose(decode_with_decay(q, k, v, 20, 1.0, recent=4), ref)
    out = decode_with_decay(q, k, v, 20, 0.9, recent=4)
    assert np.isfinite(out).all() and out.shape == ref.shape
