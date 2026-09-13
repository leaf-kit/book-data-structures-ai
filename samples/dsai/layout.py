"""8장. 정렬된 배열에 다시 서기. 배치와 분기 없는 탐색.

같은 정렬된 키를 세 방식으로 찾는다. 이진 탐색, 아이칭거 배치,
그리고 질의 배치의 정렬 병합.
아이칭거 배치는 이진 탐색이 지나는 노드를 힙 순서로 놓아
위쪽 노드가 한 라인에 모이게 한다.
"""
from __future__ import annotations

import numpy as np


def binary_search(keys: np.ndarray, k: int) -> tuple[int, int]:
    """k 이상인 첫 자리와 비교 횟수. 비교마다 분기가 있다."""
    lo, hi, steps = 0, len(keys), 0
    while lo < hi:
        mid = (lo + hi) >> 1
        steps += 1
        if keys[mid] < k:
            lo = mid + 1
        else:
            hi = mid
    return lo, steps


def eytzinger(keys: np.ndarray) -> np.ndarray:
    """정의 8.12. 정렬된 키를 힙 순서 (1 기반, 자식 2i 와 2i+1) 로 다시 놓는다."""
    n = len(keys)
    out = np.empty(n + 1, dtype=keys.dtype)
    out[0] = 0
    it = iter(keys)

    def fill(i: int) -> None:
        if i <= n:
            fill(2 * i)
            out[i] = next(it)
            fill(2 * i + 1)

    import sys
    sys.setrecursionlimit(max(1000, 4 * int(np.log2(n + 1)) + 100))
    fill(1)
    return out


def eytzinger_search(tree: np.ndarray, k: int) -> tuple[int, int]:
    """알고리즘 8.4. 분기 없는 내려가기. i 를 2i 또는 2i+1 로만 옮긴다."""
    n = len(tree) - 1
    i, steps = 1, 0
    while i <= n:
        steps += 1
        i = 2 * i + int(tree[i] < k)          # 비교 결과가 곧 자식 번호
    # 마지막으로 왼쪽으로 꺾인 자리가 답. 뒤에 붙은 1 비트 (오른쪽 꺾임) 를 걷어 낸다
    j = i >> ((~i & (i + 1)).bit_length())
    return j, steps


def batch_search(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """알고리즘 8.5. 질의를 정렬해 한 번에 찾는다.
    NumPy 의 searchsorted 는 이 꼴이다.
    """
    order = np.argsort(queries, kind="stable")
    pos = np.searchsorted(keys, queries[order], side="left")
    out = np.empty_like(pos)
    out[order] = pos
    return out
