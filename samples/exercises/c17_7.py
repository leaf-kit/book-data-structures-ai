"""문제 17.7. 규모를 바꾸며 판정이 뒤집히는 자리를 찾는다. 배열 훑기 대 해시 조회."""
import time

import numpy as np


def first_index_scan(arr: np.ndarray, key: int) -> int:
    hit = np.flatnonzero(arr == key)
    return int(hit[0]) if len(hit) else -1


def crossover(sizes):
    out = {}
    for n in sizes:
        arr = np.arange(n, dtype=np.int64)
        table = {int(v): i for i, v in enumerate(arr)}
        key = n - 1
        t0 = time.perf_counter()
        for _ in range(200):
            first_index_scan(arr, key)
        scan = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter()
        for _ in range(200):
            table[key]
        hashed = (time.perf_counter() - t0) / 200
        out[n] = (scan, hashed)
    return out


def test_scan_loses_as_n_grows():
    res = crossover([10, 1_000, 100_000])
    ratios = [res[n][0] / res[n][1] for n in (10, 1_000, 100_000)]
    assert ratios[-1] > ratios[0]                     # 커질수록 훑기가 불리하다
    assert res[100_000][0] > res[100_000][1]
