"""문제 17.5. 이 책의 실험 하나를 골라 뒷봉투 계산을 먼저 적고 실측과 견준다.
7장의 블룸 필터.
"""
import numpy as np

from dsai.bloom import BloomFilter
from dsai.choose import envelope_seconds, within_factor


def test_bloom_lookup_envelope_is_a_lower_bound():
    n, m = 100_000, 1_600_000
    bf = BloomFilter(m, 4)
    keys = np.arange(n, dtype=np.int64)
    bf.add(keys)
    import time
    t0 = time.perf_counter()
    for _ in range(3):
        bf.contains(keys)
    measured = (time.perf_counter() - t0) / 3
    # 해시 4 개가 각각 라인 하나를 당긴다. 키 n 개면 라인 4n 개, 64 바이트씩
    predicted = envelope_seconds(4 * n * 64, 4 * n * 10, 16.7, 192)
    assert measured >= predicted                      # 뒷봉투는 하한이다
    print(within_factor(predicted, measured, 3))      # 파이썬이면 3 배 밖일 수 있다
