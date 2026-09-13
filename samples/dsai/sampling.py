"""10장. 누적합과 별칭표. 확률 분포에서 뽑기.

어휘 V 개의 확률에서 하나를 뽑는 세 방법.
누적합 위의 이진 탐색 (log V), 별칭표 (상수), 그리고 top-p 절단.
언어 모델의 매 단계가 이 일이고, V 가 십만이다.
"""
from __future__ import annotations

import numpy as np


def sample_cumsum(p: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """정의 10.6. 누적합 C 를 만들고 u ~ U(0,1) 마다 C 에서 u 이상인 첫 자리.
    8장의 이진 탐색.
    """
    c = np.cumsum(p)
    u = rng.random(n) * c[-1]
    return np.searchsorted(c, u, side="right")


class AliasTable:
    """정의 10.7. 자리 V 개마다 확률 하나와 별칭 하나.
    자리를 고르고 동전을 던지면 끝난다.
    """

    def __init__(self, p: np.ndarray):
        n = len(p)
        q = p / p.sum() * n
        self.prob = np.zeros(n)
        self.alias = np.zeros(n, dtype=np.int64)
        small = [i for i in range(n) if q[i] < 1.0]
        large = [i for i in range(n) if q[i] >= 1.0]
        while small and large:                  # 알고리즘 10.3. Vose 의 만들기
            s, l = small.pop(), large.pop()
            self.prob[s] = q[s]
            self.alias[s] = l
            q[l] = q[l] + q[s] - 1.0
            (small if q[l] < 1.0 else large).append(l)
        for i in small + large:
            self.prob[i] = 1.0
            self.alias[i] = i

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """뽑기 하나에 무작위 둘과 읽기 둘. V 와 무관하다."""
        i = rng.integers(0, len(self.prob), n)
        coin = rng.random(n) < self.prob[i]
        return np.where(coin, i, self.alias[i])


def top_p_mask(p: np.ndarray, top_p: float) -> np.ndarray:
    """정의. top-p.
    확률을 내림차순으로 정렬해 누적이 top_p 를 넘는 첫 자리까지 남긴다.
    """
    order = np.argsort(-p)
    c = np.cumsum(p[order])
    keep = c - p[order] < top_p          # 자기 자신을 더하기 전 누적이 문턱 아래
    mask = np.zeros(len(p), dtype=bool)
    mask[order[keep]] = True
    return mask


def top_p_mask_by_select(p: np.ndarray, top_p: float, k_guess: int) -> np.ndarray:
    """10.1 절의 선택으로 top-p. 상위 k 개만 부분 정렬하고 모자라면 두 배로."""
    n = len(p)
    k = min(k_guess, n)
    while True:
        idx = np.argpartition(-p, k - 1)[:k]
        order = idx[np.argsort(-p[idx])]
        c = np.cumsum(p[order])
        if c[-1] >= top_p or k == n:
            keep = c - p[order] < top_p
            mask = np.zeros(n, dtype=bool)
            mask[order[keep]] = True
            return mask
        k = min(2 * k, n)
