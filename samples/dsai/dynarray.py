"""2장. 동적 배열. 크기가 자라는 값 배열과 그 대가.

용량을 미리 잡아 두고 넘치면 더 큰 버퍼로 옮긴다.
옮기는 값을 어떻게 나눠 갚는지가 분할 상환 분석이다.
"""
from __future__ import annotations

import numpy as np


class DynArray:
    """성장 인자 factor 로 자라는 값 배열. 옮긴 바이트를 센다."""

    def __init__(self, dtype=np.float32, factor: float = 2.0, capacity: int = 4,
                 increment: int = 0):
        self.buf = np.empty(capacity, dtype=dtype)
        self.size = 0
        self.factor = factor
        self.increment = increment  # 0 이면 곱하고, 아니면 더한다
        self.copied_bytes = 0
        self.grow_count = 0

    @property
    def capacity(self) -> int:
        return len(self.buf)

    def _grow(self) -> None:
        if self.increment:
            new_cap = self.capacity + self.increment
        else:
            new_cap = max(self.capacity + 1, int(self.capacity * self.factor))
        new_buf = np.empty(new_cap, dtype=self.buf.dtype)
        new_buf[:self.size] = self.buf[:self.size]
        self.copied_bytes += self.size * self.buf.itemsize
        self.grow_count += 1
        self.buf = new_buf

    def append(self, value) -> None:
        """분할 상환 O(1). 용량이 차면 먼저 자란다."""
        if self.size == self.capacity:
            self._grow()
        self.buf[self.size] = value
        self.size += 1

    def waste_bytes(self) -> int:
        """예약했지만 쓰지 않은 바이트. 정의 2.2 의 내부 단편화다."""
        return (self.capacity - self.size) * self.buf.itemsize

    def view(self) -> np.ndarray:
        return self.buf[:self.size]


def total_copied(n: int, factor: float, capacity: int = 4) -> int:
    """원소 n 개를 넣을 때 옮겨진 원소 수의 합. 정리 2.1 의 식을 그대로 센다."""
    cap, copied = capacity, 0
    while cap < n:
        copied += cap
        cap = max(cap + 1, int(cap * factor))
    return copied
