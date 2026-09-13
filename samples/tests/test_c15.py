import numpy as np

from dsai.attention import (attention_naive, attention_tiled, traffic_naive,
                            traffic_tiled)
from dsai.kvevict import decode_with_policy, full_attention_causal
from dsai.ssm import run_recurrent, run_scan


def test_tiled_attention_matches_naive():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((300, 32)) for _ in range(3))
    ref = attention_naive(q, k, v)
    for block in (7, 64, 300, 1000):
        assert np.allclose(attention_tiled(q, k, v, block), ref, atol=1e-9)
    assert traffic_tiled(300, 300, 32, 32, 4, 64) < traffic_naive(300, 300, 32, 32, 4)


def test_full_policy_equals_causal_attention():
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal((64, 16)) for _ in range(3))
    out, kept = decode_with_policy(q, k, v, 64, "full")
    assert np.allclose(out, full_attention_causal(q, k, v), atol=1e-9)
    assert kept.max() == 64
    out_lru, kept_lru = decode_with_policy(q, k, v, 16, "lru")
    assert kept_lru.max() == 17 and out_lru.shape == out.shape
    out_h, _ = decode_with_policy(q, k, v, 16, "heavy", recent=4)
    assert np.isfinite(out_h).all()


def test_scan_equals_recurrence():
    rng = np.random.default_rng(2)
    a = np.exp(-rng.uniform(0.01, 0.5, 8))
    b, c = rng.standard_normal(8), rng.standard_normal(8)
    x = rng.standard_normal(1000)
    y1 = run_recurrent(x, a, b, c)
    assert np.allclose(run_scan(x, a, b, c), y1, atol=1e-9)
    assert np.allclose(run_scan(x[:1], a, b, c), y1[:1])
