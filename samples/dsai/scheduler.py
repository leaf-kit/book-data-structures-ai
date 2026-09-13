"""10장. 우선순위 스케줄러. 요청을 줄 세우기.

5장의 연속 배칭은 도착 순서 (FIFO) 였다. 여기서는 힙으로 우선순위를 둔다.
짧은 것 먼저 (SJF) 는 평균 대기를 줄이고, 긴 요청을 굶긴다. 나이를 더하면 굶김이 는다.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Item:
    key: float
    arrival: float = field(compare=False)
    length: int = field(compare=False)
    rid: int = field(compare=False)


class PriorityQueue:
    """정의 10.10. 키가 가장 작은 항목을 먼저 내는 큐. 힙 하나다."""

    def __init__(self):
        self.h: list[Item] = []

    def push(self, item: Item) -> None:
        heapq.heappush(self.h, item)

    def pop(self) -> Item:
        return heapq.heappop(self.h)

    def __len__(self) -> int:
        return len(self.h)


def simulate(arrivals, policy: str, rate: float, aging: float = 0.0):
    """알고리즘 10.5. 요청 (도착, 길이) 를 정책대로 처리한다. 처리량 rate 토큰/초,
    한 번에 하나.

    policy: fifo | sjf | aged  (aged 는 길이에서 기다린 시간 x aging 을 뺀다)
    대기 시간의 평균과 최대를 돌려준다.
    """
    q = PriorityQueue()
    waits = []
    t = 0.0
    i = 0
    n = len(arrivals)
    while i < n or len(q):
        while i < n and arrivals[i][0] <= t:
            a, L = arrivals[i]
            key = a if policy == "fifo" else float(L)
            q.push(Item(key, a, L, i))
            i += 1
        if not len(q):
            t = arrivals[i][0]
            continue
        if policy == "aged":                          # 키를 나이로 다시 매긴다
            items = q.h
            q.h = []
            for it in items:
                it.key = it.length - aging * (t - it.arrival)
                q.push(it)
        it = q.pop()
        waits.append(t - it.arrival)
        t += it.length / rate
    waits.sort()
    return sum(waits) / len(waits), waits[-1], waits[int(0.99 * (len(waits) - 1))]
