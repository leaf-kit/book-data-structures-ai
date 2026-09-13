"""문제 10.8. 나이 정책을 다시 매기지 않고. 키 = 길이 + alpha x 도착 시각."""
import numpy as np

from dsai.scheduler import Item, PriorityQueue, simulate


def simulate_static_aging(arrivals, rate, aging):
    q = PriorityQueue()
    waits, t, i, n = [], 0.0, 0, len(arrivals)
    while i < n or len(q):
        while i < n and arrivals[i][0] <= t:
            a, L = arrivals[i]
            q.push(Item(L + aging * a, a, L, i))      # 넣을 때 한 번만 정한다
            i += 1
        if not len(q):
            t = arrivals[i][0]
            continue
        it = q.pop()
        waits.append(t - it.arrival)
        t += it.length / rate
    waits.sort()
    return sum(waits) / len(waits), waits[-1]


def test_static_aging_equals_rekeying():
    rng = np.random.default_rng(3)
    lengths = np.minimum(rng.zipf(1.6, 1500) * 32, 4096)
    arr = list(zip(np.cumsum(rng.exponential(0.06, 1500)).tolist(), lengths.tolist()))
    a = simulate(arr, "aged", 2000.0, aging=200.0)
    b = simulate_static_aging(arr, 2000.0, 200.0)
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
