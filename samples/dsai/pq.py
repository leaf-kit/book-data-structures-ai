"""13장. 곱 양자화. 벡터를 바이트 몇 개로.

벡터를 부분 공간 m 개로 자르고, 부분마다 중심 256 개의 코드북을 둔다.
벡터 하나가 바이트 m 개가 된다.
거리는 부분마다 (질의 부분, 중심) 거리표를 한 번 만들고,
바이트를 첨자로 표를 읽어 더한다. 비대칭 거리다.
"""
from __future__ import annotations

import numpy as np

from dsai.ivf import assign, kmeans


class PQ:
    """정의 13.3. 부분 m 개, 부분마다 코드북 (256, d/m). 코드는 (n, m) uint8."""

    def __init__(self, x: np.ndarray, m: int, rng: np.random.Generator,
                 iters: int = 10):
        n, d = x.shape
        assert d % m == 0
        self.m, self.ds = m, d // m
        parts = [x[:, j * self.ds:(j + 1) * self.ds] for j in range(m)]
        self.books = np.stack([kmeans(p, 256, rng, iters) for p in parts])

    def encode(self, x: np.ndarray) -> np.ndarray:
        codes = np.empty((len(x), self.m), dtype=np.uint8)
        for j in range(self.m):
            codes[:, j] = assign(x[:, j * self.ds:(j + 1) * self.ds], self.books[j])
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        parts = [self.books[j][codes[:, j]] for j in range(self.m)]
        return np.concatenate(parts, axis=1)

    def table(self, q: np.ndarray) -> np.ndarray:
        """알고리즘 13.3. 질의 하나의 거리표 (m, 256).
        부분마다 질의 조각과 중심 256 개의 거리.
        """
        t = np.empty((self.m, 256))
        for j in range(self.m):
            qj = q[j * self.ds:(j + 1) * self.ds]
            t[j] = ((self.books[j] - qj) ** 2).sum(axis=1)
        return t

    def adc(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """비대칭 거리. 코드 (n, m) 마다 표를 m 번 읽어 더한다.
        벡터를 복원하지 않는다.
        """
        return table[np.arange(self.m)[None, :], codes].sum(axis=1)

    def nbytes_per_vector(self) -> int:
        return self.m


def quantization_error(pq: PQ, x: np.ndarray) -> float:
    """정의 13.3 의 재구성 오차.
    |x - decode(encode(x))|^2 의 평균을 |x|^2 의 평균으로 나눈 것.
    """
    r = pq.decode(pq.encode(x))
    return float(((x - r) ** 2).sum(axis=1).mean() / (x ** 2).sum(axis=1).mean())
