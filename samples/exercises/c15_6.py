"""문제 15.6. 타일 크기를 바꾸며 시간을 잰다. 작으면 호출 값,
크면 블록 점수가 캐시를 넘친다.
"""
import time

import numpy as np

from dsai.attention import attention_naive, attention_tiled


def sweep(q, k, v, blocks):
    times = {}
    for b in blocks:
        t0 = time.perf_counter()
        out = attention_tiled(q, k, v, b)
        times[b] = time.perf_counter() - t0
        assert np.allclose(out, attention_naive(q, k, v), atol=1e-9)
    return times


def test_all_blocks_agree_and_a_minimum_exists():
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal((2048, 64)) for _ in range(3))
    times = sweep(q, k, v, [8, 64, 512, 2048])
    best = min(times, key=times.get)
    assert best in times and all(t > 0 for t in times.values())
