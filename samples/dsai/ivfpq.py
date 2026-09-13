"""13장. 역파일과 곱 양자화의 결합. 군집으로 후보를 줄이고,
잔차의 코드로 거리를 어림한다.

점을 군집 중심에서 뺀 잔차를 곱 양자화한다.
잔차는 원래 벡터보다 작아 같은 바이트로 오차가 작다.
질의는 군집마다 잔차 거리표를 만들고 그 군집의 코드에 비대칭 거리를 더한다.
"""
from __future__ import annotations

import numpy as np

from dsai.ivf import IVF
from dsai.pq import PQ


class IVFPQ:
    """정의 13.4. 역파일 + 잔차의 곱 양자화. 저장은 군집 순서의 코드 (n, m) 과 번호."""

    def __init__(self, x: np.ndarray, centroids: np.ndarray, m: int,
                 rng: np.random.Generator):
        self.ivf = IVF(x, centroids)
        a = np.repeat(np.arange(len(centroids)), np.diff(self.ivf.ptr))
        resid = self.ivf.xs - centroids[a]                # 군집 순서의 잔차
        self.pq = PQ(resid, m, rng)
        self.codes = self.pq.encode(resid)                # (n, m) uint8, 군집 순서
        self.c = centroids

    def search(self, q: np.ndarray, nprobe: int, k: int,
               rerank: int = 0) -> tuple[np.ndarray, int]:
        """알고리즘 13.4. 가까운 군집마다 잔차 거리표를 만들고 코드에 더한다.
        rerank 개는 정확히 다시 잰다.
        """
        dc = ((self.c - q) ** 2).sum(axis=1)
        lists = np.argpartition(dc, nprobe - 1)[:nprobe]
        cand, dist = [], []
        for l in lists:
            lo, hi = self.ivf.ptr[l], self.ivf.ptr[l + 1]
            if hi == lo:
                continue
            t = self.pq.table(q - self.c[l])
            cand.append(np.arange(lo, hi))
            dist.append(self.pq.adc(t, self.codes[lo:hi]))
        cand, dist = np.concatenate(cand), np.concatenate(dist)
        if rerank:
            top = cand[np.argpartition(dist, min(rerank, len(cand)) - 1)[:rerank]]
            exact = ((self.ivf.xs[top] - q) ** 2).sum(axis=1)
            top = top[np.argsort(exact)[:k]]
        else:
            top = cand[np.argsort(dist)[:k]]
        return self.ivf.ids[top], len(cand)

    def nbytes(self) -> int:
        return (self.codes.nbytes + self.ivf.ids.nbytes + self.ivf.ptr.nbytes
                + self.pq.books.nbytes)
