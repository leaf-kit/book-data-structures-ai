"""문제 8.8. 정렬된 질의를 앞 질의의 자리에서 갤로핑으로 잇는다."""
import numpy as np


def gallop_from(keys: np.ndarray, k: int, start: int) -> int:
    """start 부터 보폭을 두 배씩 늘려 k 이상인 첫 자리를 찾는다."""
    n = len(keys)
    if start >= n or keys[start] >= k:
        return start
    step, lo = 1, start
    hi = min(start + step, n)
    while hi < n and keys[hi] < k:
        lo = hi
        step *= 2
        hi = min(start + step, n)
    return lo + int(np.searchsorted(keys[lo:hi], k))


def gallop_batch(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    order = np.argsort(queries, kind="stable")
    out = np.empty(len(queries), dtype=np.int64)
    pos = 0
    for i in order:
        pos = gallop_from(keys, int(queries[i]), pos)
        out[i] = pos
    return out


def test_gallop_matches_searchsorted():
    rng = np.random.default_rng(7)
    keys = np.sort(rng.choice(1 << 30, 100_000, replace=False))
    q = rng.choice(1 << 30, 3000)
    assert np.array_equal(gallop_batch(keys, q), np.searchsorted(keys, q))
