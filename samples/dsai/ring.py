"""5장. 링 버퍼. 고정 크기 배열 위의 큐.

머리와 꼬리 번호 둘이 배열을 돌면서 앞에서 빼고 뒤에 넣는다. 옮기는 원소가 없다.
가장 오래된 것이 자동으로 덮이는 창이 되고, 그것이 슬라이딩 윈도우다.
"""
from __future__ import annotations

import numpy as np


class RingBuffer:
    """정의 5.3. 용량 c 의 배열과 머리 h, 개수 n. 꼬리는 (h + n) mod c 다."""

    def __init__(self, capacity: int, dtype=np.float64):
        self.buf = np.zeros(capacity, dtype=dtype)
        self.cap = capacity
        self.head = 0
        self.n = 0

    def push(self, v) -> None:
        """꼬리에 넣는다. 차 있으면 머리의 것을 덮는다."""
        tail = (self.head + self.n) % self.cap
        self.buf[tail] = v
        if self.n == self.cap:
            self.head = (self.head + 1) % self.cap
        else:
            self.n += 1

    def pop(self):
        """머리에서 뺀다. 옮기는 것 없이 번호 하나만 움직인다."""
        v = self.buf[self.head]
        self.head = (self.head + 1) % self.cap
        self.n -= 1
        return v

    def view(self) -> np.ndarray:
        """논리 순서. 두 조각을 이어 붙인다. 이것만 복사가 든다."""
        end = self.head + self.n
        if end <= self.cap:
            return self.buf[self.head:end]
        return np.concatenate([self.buf[self.head:], self.buf[:end - self.cap]])


class WindowMean:
    """길이 w 의 창 평균을 갱신마다 상수 시간에. 링 버퍼와 누적합 하나."""

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, v: float) -> float:
        if self.ring.n == self.ring.cap:
            self.total -= self.ring.buf[self.ring.head]
        self.ring.push(v)
        self.total += v
        return self.total / self.ring.n


def window_mean_recompute(xs: np.ndarray, w: int) -> np.ndarray:
    """창을 매번 다시 더한다. 갱신마다 w 개를 읽는다."""
    out = np.empty(len(xs))
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out[i] = xs[lo:i + 1].mean()
    return out
