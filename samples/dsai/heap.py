"""10장. 이진 힙과 top-k. 전부 정렬하지 않는 값.

힙은 배열 하나이고 부모가 자식보다 크다는 불변식만 지킨다. 최대를 꺼내는 데 로그,
전부 정렬은 안 한다.
상위 k 개를 고르는 데 힙 크기 k 를 두면 n log k 이고, 부분 정렬 (선택) 은 n 이다.
"""
from __future__ import annotations

import numpy as np


class MaxHeap:
    """정의 10.1. 배열 위의 완전 이진 트리. 자식 2i+1,
    2i+2 이고 부모가 자식 이상이다.
    """

    def __init__(self):
        self.a: list[float] = []

    def push(self, x: float) -> None:
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i > 0:
            p = (i - 1) >> 1
            if a[p] >= a[i]:
                break
            a[p], a[i] = a[i], a[p]
            i = p

    def pop(self) -> float:
        a = self.a
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            self._sift_down(0)
        return top

    def _sift_down(self, i: int) -> None:
        a, n = self.a, len(self.a)
        while True:
            l, r, big = 2 * i + 1, 2 * i + 2, i
            if l < n and a[l] > a[big]:
                big = l
            if r < n and a[r] > a[big]:
                big = r
            if big == i:
                return
            a[i], a[big] = a[big], a[i]
            i = big

    def __len__(self) -> int:
        return len(self.a)


def topk_heap(x: np.ndarray, k: int) -> np.ndarray:
    """알고리즘 10.1. 크기 k 의 최소 힙을 유지한다.
    원소마다 힙의 최소와 견줘 크면 바꾼다. n log k.
    """
    heap = MaxHeap()
    for v in x[:k]:
        heap.push(-float(v))            # 최대 힙에 부호를 바꿔 넣어 최소 힙으로 쓴다
    for v in x[k:]:
        if -heap.a[0] < v:
            heap.a[0] = -float(v)
            heap._sift_down(0)
    return -np.array(heap.a)


def topk_select(x: np.ndarray, k: int) -> np.ndarray:
    """명제 10.3. 부분 정렬. k 번째로 큰 값을 기준으로 앞뒤를 가른다. 기대 n."""
    idx = np.argpartition(-x, k - 1)[:k]
    return x[idx]


def topk_sort(x: np.ndarray, k: int) -> np.ndarray:
    """디딤돌. 전부 정렬하고 앞 k 개. n log n."""
    return np.sort(x)[::-1][:k]
