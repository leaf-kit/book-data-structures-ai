"""문제 4.5. 같이 가는 고리 수를 바꿔 가며 걸음당 시간을 재고 포화점을 찾는다."""
import numpy as np

from dsai.linked import chain_random, chase_many


def per_step_ns(nxt: np.ndarray, k: int, total_steps: int, rng) -> float:
    import time
    starts = rng.integers(0, len(nxt), k)
    steps = max(1, total_steps // k)
    t0 = time.perf_counter()
    chase_many(nxt, starts, steps)
    return (time.perf_counter() - t0) * 1e9 / (steps * k)


def test_more_chains_less_time_per_step():
    rng = np.random.default_rng(1)
    nxt = chain_random(1 << 20, 2)
    t1 = per_step_ns(nxt, 1, 20_000, rng)
    t256 = per_step_ns(nxt, 256, 20_000, rng)
    assert t256 < t1
