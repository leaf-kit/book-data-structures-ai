"""4장. 연결 리스트. 포인터를 따라가는 값과, 그 값을 배열 안에서 치르는 법.

노드 하나가 다음 노드의 주소를 든다.
순회는 주소를 읽고 그 주소로 가는 일을 n 번 반복한다.
읽기 하나가 끝나야 다음 주소를 알므로, 프로세서가 미리 가져올 수 없다.
"""
from __future__ import annotations

import numpy as np


def chain_sequential(n: int) -> np.ndarray:
    """next[i] = i + 1. 메모리 순서와 순회 순서가 같다."""
    nxt = np.arange(1, n + 1, dtype=np.int64)
    nxt[-1] = 0
    return nxt


def chain_random(n: int, seed: int) -> np.ndarray:
    """무작위 순열을 따라 하나의 큰 고리를 만든다. 메모리 순서와 순회 순서가 다르다."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    nxt = np.empty(n, dtype=np.int64)
    nxt[order[:-1]] = order[1:]
    nxt[order[-1]] = order[0]
    return nxt


def chase(nxt: np.ndarray, start: int, steps: int) -> int:
    """정의 4.1 의 순회. 읽기 하나가 다음 읽기의 주소를 정한다."""
    i = start
    for _ in range(steps):
        i = nxt[i]
    return int(i)


def chase_many(nxt: np.ndarray, starts: np.ndarray, steps: int) -> np.ndarray:
    """독립된 고리 k 개를 한 걸음씩 같이 간다. 읽기 k 개가 서로를 기다리지 않는다."""
    idx = starts.copy()
    for _ in range(steps):
        idx = nxt[idx]
    return idx


def traverse_sum(values: np.ndarray, nxt: np.ndarray, start: int, count: int) -> float:
    """리스트를 count 개 따라가며 값을 더한다. 배열의 sum 과 같은 답, 다른 이동량."""
    total = 0.0
    i = start
    for _ in range(count):
        total += values[i]
        i = nxt[i]
    return total


class IndexList:
    """정의 4.3. 배열 안의 연결 리스트. 노드는 배열의 칸이고 포인터는 칸 번호다."""

    def __init__(self, capacity: int, dtype=np.float32):
        self.val = np.empty(capacity, dtype=dtype)
        self.nxt = np.full(capacity, -1, dtype=np.int64)
        self.head = -1
        self.tail = -1
        self.size = 0
        self.free = list(range(capacity - 1, -1, -1))   # 자유 리스트. 4.2 절

    def append(self, v) -> int:
        i = self.free.pop()
        self.val[i] = v
        self.nxt[i] = -1
        if self.tail >= 0:
            self.nxt[self.tail] = i
        else:
            self.head = i
        self.tail = i
        self.size += 1
        return i

    def insert_after(self, i: int, v) -> int:
        """칸 i 뒤에 끼운다. 배열이면 뒤를 전부 밀어야 하는 연산이 상수다."""
        j = self.free.pop()
        self.val[j] = v
        self.nxt[j] = self.nxt[i]
        self.nxt[i] = j
        if self.tail == i:
            self.tail = j
        self.size += 1
        return j

    def order(self) -> np.ndarray:
        """순회 순서대로 칸 번호를 늘어놓는다."""
        out = np.empty(self.size, dtype=np.int64)
        i, k = self.head, 0
        while i >= 0:
            out[k] = i
            i = self.nxt[i]
            k += 1
        return out

    def total(self) -> float:
        return traverse_sum(self.val, self.nxt, self.head, self.size)

    def gather_sum(self, order: np.ndarray) -> float:
        """순회 순서를 인덱스 배열로 받아 한 번에 모아 더한다.
        이동량은 순서가 정한다.
        """
        return float(self.val[order].sum())

    def compact(self) -> None:
        """알고리즘 4.2. 순회 순서대로 칸을 다시 놓는다. 그 뒤 순회는 순차 접근이다."""
        order = self.order()
        self.val[:self.size] = self.val[order]
        self.nxt[:self.size - 1] = np.arange(1, self.size, dtype=np.int64)
        self.nxt[self.size - 1] = -1
        self.head, self.tail = 0, self.size - 1
        self.free = list(range(len(self.val) - 1, self.size - 1, -1))
