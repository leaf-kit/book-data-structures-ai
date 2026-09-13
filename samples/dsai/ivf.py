"""13장. k-평균과 역파일. 점을 군집으로 나누고 가까운 군집만 본다.

k-평균은 중심 k 개를 두고 배정과 갱신을 되풀이한다.
역파일은 군집마다 점의 번호 목록이고, 11장의 CSR 이다.
질의는 가까운 중심 nprobe 개를 고르고 그 목록의 점만 견준다.
"""
from __future__ import annotations

import numpy as np


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator,
           iters: int = 10) -> np.ndarray:
    """알고리즘 13.1. Lloyd 의 반복. 중심 (k, d)."""
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        a = assign(x, c)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(axis=0)
    return c


def assign(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """가장 가까운 중심의 번호. |x|^2 - 2 x.c + |c|^2 를 행렬 곱으로."""
    d2 = (x * x).sum(axis=1)[:, None] - 2.0 * x @ c.T + (c * c).sum(axis=1)[None, :]
    return np.argmin(d2, axis=1)


class IVF:
    """정의 13.2. 역파일. 군집마다 점 번호의 연속 구간. 포인터 하나와 번호 배열 하나."""

    def __init__(self, x: np.ndarray, centroids: np.ndarray):
        self.x = x
        self.c = centroids
        a = assign(x, centroids)
        order = np.argsort(a, kind="stable")
        self.ids = order.astype(np.int64)
        counts = np.bincount(a, minlength=len(centroids))
        self.ptr = np.zeros(len(centroids) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.ptr[1:])
        self.xs = x[self.ids]                  # 군집 순서로 다시 놓은 벡터

    def search(self, q: np.ndarray, nprobe: int, k: int) -> tuple[np.ndarray, int]:
        """알고리즘 13.2. 가까운 중심 nprobe 개의 목록만 견준다. (상위 k 번호,
        견준 점 수).
        """
        dc = ((self.c - q) ** 2).sum(axis=1)
        lists = np.argpartition(dc, nprobe - 1)[:nprobe]
        cand = np.concatenate([np.arange(self.ptr[l], self.ptr[l + 1]) for l in lists])
        d2 = ((self.xs[cand] - q) ** 2).sum(axis=1)
        top = cand[np.argsort(d2)[:k]]
        return self.ids[top], len(cand)
