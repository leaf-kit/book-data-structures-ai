"""7장. 블룸 필터. 있음은 틀릴 수 있고 없음은 안 틀리는 집합.

비트 m 개와 해시 k 개. 넣을 때 k 자리를 1 로 만들고,
물을 때 k 자리가 전부 1 이면 있다고 답한다.
키를 저장하지 않으므로 키 하나에 몇 비트면 된다.
"""
from __future__ import annotations

import math

import numpy as np

from dsai.hashing import mix64


def hashes(keys: np.ndarray, k: int, m: int) -> np.ndarray:
    """키마다 자리 k 개. 해시 둘로 k 개를 만드는 이중 해싱. (n, k) 배열."""
    h1 = mix64(keys.astype(np.uint64))
    h2 = mix64(h1 ^ np.uint64(0x5851F42D4C957F2D)) | np.uint64(1)
    i = np.arange(k, dtype=np.uint64)
    return ((h1[:, None] + i[None, :] * h2[:, None]) % np.uint64(m)).astype(np.int64)


class BloomFilter:
    """정의 7.1. 비트 배열과 해시 k 개."""

    def __init__(self, m_bits: int, k: int):
        self.m = m_bits
        self.k = k
        self.bits = np.zeros(m_bits, dtype=bool)

    def add(self, keys: np.ndarray) -> None:
        self.bits[hashes(keys, self.k, self.m).ravel()] = True

    def contains(self, keys: np.ndarray) -> np.ndarray:
        """자리 k 개가 전부 1 인가. 없는 키도 우연히 전부 1 이면 있다고 답한다."""
        return self.bits[hashes(keys, self.k, self.m)].all(axis=1)

    def nbytes(self) -> int:
        return self.m // 8


def false_positive_rate(m: int, n: int, k: int) -> float:
    """정리 7.2. 키 n 개를 넣은 뒤 없는 키가 있다고 나올 확률의 근사."""
    return (1.0 - math.exp(-k * n / m)) ** k


def optimal_k(m: int, n: int) -> int:
    """비트 수와 키 수가 정해졌을 때 오답을 가장 작게 하는 해시 개수. (m/n) ln 2."""
    return max(1, round(m / n * math.log(2)))


def bits_per_key_for(fpr: float) -> float:
    """목표 오답률에 필요한 키당 비트. -log2(fpr) / ln 2."""
    return -math.log2(fpr) / math.log(2)
