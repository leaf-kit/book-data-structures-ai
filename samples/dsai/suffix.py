"""14장. 접미사 배열. 텍스트의 모든 접미사를 정렬한 번호 배열.

접미사 배열 위에서 패턴 찾기는 이진 탐색이고,
같은 접두사를 가진 접미사들이 연속 구간이다.
그 구간의 길이가 n-그램의 개수다.
만들기는 배가 정렬 (prefix doubling) 이고 정렬 log n 번이다.
"""
from __future__ import annotations

import numpy as np


def build_suffix_array(text: np.ndarray) -> np.ndarray:
    """알고리즘 14.1. 배가 정렬. 길이 2^k 접두사의 순위로 정렬을 되풀이한다.
    O(n log^2 n).
    """
    n = len(text)
    _, rank = np.unique(text, return_inverse=True)      # 조밀한 순위 0..r 로 시작
    rank = rank.astype(np.int64)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        second[: n - k] = rank[k:]
        key = rank * (n + 1) + (second + 1)          # (앞 순위, 뒤 순위) 를 하나의 키로
        sa = np.argsort(key, kind="stable")
        sorted_key = key[sa]
        new_rank = np.zeros(n, dtype=np.int64)
        new_rank[sa[1:]] = np.cumsum(sorted_key[1:] != sorted_key[:-1])
        rank = new_rank
        if rank.max() == n - 1:
            break
        k *= 2
    return sa


def lcp_array(text: np.ndarray, sa: np.ndarray) -> np.ndarray:
    """Kasai 의 방법. 이웃한 접미사의 공통 접두사 길이. lcp[i] = lcp(sa[i-1], sa[i]).
    O(n).
    """
    n = len(text)
    rank = np.empty(n, dtype=np.int64)
    rank[sa] = np.arange(n)
    lcp = np.zeros(n, dtype=np.int64)
    h = 0
    t = text.tolist()
    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and t[i + h] == t[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return lcp


def find_range(text: np.ndarray, sa: np.ndarray,
               pattern: np.ndarray) -> tuple[int, int]:
    """알고리즘 14.2. 패턴으로 시작하는 접미사의 구간 [lo, hi). O(|P| log n)."""
    n, m = len(text), len(pattern)
    pat = pattern.tolist()

    def less(i: int, strict: bool) -> bool:
        """접미사 sa[i] 의 앞 m 글자가 패턴보다 작은가. strict 면 같아도 작다."""
        s = text[sa[i]: sa[i] + m].tolist()
        for a, b in zip(s, pat):
            if a != b:
                return a < b
        return len(s) < m or strict

    def bound(strict: bool) -> int:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if less(mid, strict):
                lo = mid + 1
            else:
                hi = mid
        return lo

    return bound(False), bound(True)


def count(text: np.ndarray, sa: np.ndarray, pattern: np.ndarray) -> int:
    lo, hi = find_range(text, sa, pattern)
    return hi - lo
