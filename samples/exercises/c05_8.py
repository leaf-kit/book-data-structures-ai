"""문제 5.8. 대기 큐를 짧은 요청 먼저로 바꾼 연속 배칭."""
import numpy as np

from dsai.batching import Request, simulate_continuous, summarize


def simulate_sjf(reqs: list[Request], max_batch: int, fixed_ms: float,
                 per_token_ms: float) -> None:
    """도착한 요청 중 가장 짧은 것을 먼저 넣는다."""
    from dsai.batching import step_time
    pending = sorted(reqs, key=lambda r: r.arrival)
    active: list[tuple[Request, int]] = []
    clock = 0.0
    while pending or active:
        if not active and pending:
            clock = max(clock, pending[0].arrival)
        arrived = [r for r in pending if r.arrival <= clock]
        arrived.sort(key=lambda r: r.tokens)
        while arrived and len(active) < max_batch:
            r = arrived.pop(0)
            pending.remove(r)
            active.append((r, 0))
        clock += step_time(len(active), fixed_ms, per_token_ms)
        still = []
        for r, done in active:
            done += 1
            if done == r.tokens:
                r.done_at = clock
            else:
                still.append((r, done))
        active = still


def test_sjf_lowers_mean_latency():
    rng = np.random.default_rng(9)
    arr = np.cumsum(rng.exponential(250, 300))
    tok = np.clip(rng.lognormal(4.0, 0.7, 300).astype(int), 4, 512)
    a = [Request(float(x), int(t)) for x, t in zip(arr, tok)]
    b = [Request(float(x), int(t)) for x, t in zip(arr, tok)]
    simulate_continuous(a, 8, 20.0, 0.5)
    simulate_sjf(b, 8, 20.0, 0.5)
    assert summarize(b)[1] <= summarize(a)[1]
