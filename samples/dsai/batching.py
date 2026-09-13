"""5장. 배칭 큐. 요청을 모아 한 번에 처리하는 큐와, 토큰 단위로 갈아 끼우는 큐.

정적 배칭은 배치가 찰 때까지 기다렸다가 배치 안의 모든 요청이 끝날 때까지 같이 간다.
연속 배칭은 토큰 하나를 낼 때마다 끝난 요청을 빼고 기다리는 요청을 넣는다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Request:
    arrival: float
    tokens: int          # 내야 하는 토큰 수
    done_at: float = -1.0


def step_time(batch_size: int, fixed_ms: float, per_token_ms: float) -> float:
    """토큰 한 걸음의 시간. 가중치를 읽는 고정 값과 토큰마다의 값."""
    return fixed_ms + per_token_ms * batch_size


def simulate_static(reqs: list[Request], max_batch: int, fixed_ms: float,
                    per_token_ms: float) -> None:
    """알고리즘 5.3 의 앞 절반. 배치가 차거나 큐가 비면 시작하고,
    가장 긴 요청이 끝날 때까지 간다.
    """
    queue = deque(sorted(reqs, key=lambda r: r.arrival))
    clock = 0.0
    while queue:
        clock = max(clock, queue[0].arrival)
        batch = []
        while queue and len(batch) < max_batch and queue[0].arrival <= clock:
            batch.append(queue.popleft())
        longest = max(r.tokens for r in batch)
        for step in range(longest):
            clock += step_time(len(batch), fixed_ms, per_token_ms)
            for r in batch:
                if r.tokens == step + 1:
                    r.done_at = clock


def simulate_continuous(reqs: list[Request], max_batch: int, fixed_ms: float,
                        per_token_ms: float) -> None:
    """알고리즘 5.3 의 뒤 절반. 걸음마다 끝난 요청을 빼고 기다리는 요청을 넣는다."""
    queue = deque(sorted(reqs, key=lambda r: r.arrival))
    active: list[tuple[Request, int]] = []
    clock = 0.0
    while queue or active:
        if not active and queue:
            clock = max(clock, queue[0].arrival)
        while queue and len(active) < max_batch and queue[0].arrival <= clock:
            active.append((queue.popleft(), 0))
        clock += step_time(len(active), fixed_ms, per_token_ms)
        still = []
        for r, done in active:
            done += 1
            if done == r.tokens:
                r.done_at = clock
            else:
                still.append((r, done))
        active = still


def summarize(reqs: list[Request]) -> tuple[float, float, float]:
    """처리량 (토큰/초), 평균 지연 (ms), 95 퍼센타일 지연 (ms)."""
    import numpy as np
    lat = np.array([r.done_at - r.arrival for r in reqs])
    total_tokens = sum(r.tokens for r in reqs)
    makespan = max(r.done_at for r in reqs) - min(r.arrival for r in reqs)
    throughput = total_tokens / (makespan / 1000)
    return throughput, float(lat.mean()), float(np.percentile(lat, 95))
