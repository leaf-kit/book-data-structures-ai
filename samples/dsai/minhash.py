"""7장. MinHash 와 지역 민감 해싱. 비슷한 것끼리 같은 자리에 오게 하는 해시.

집합 둘의 자카드 유사도는 교집합을 합집합으로 나눈 것이다.
무작위 순서에서 각 집합의 최소 원소가 같을 확률이 정확히 그 값이다.
그래서 최소값 k 개가 서명이 된다.
서명을 띠로 나눠 해시하면 비슷한 것끼리만 같은 자리에 모인다.
"""
from __future__ import annotations

import numpy as np

from dsai.hashing import mix64


def shingles(text: str, n: int = 3) -> set[int]:
    """문자 n 개짜리 조각의 집합. 문서를 집합으로 바꾸는 가장 단순한 방법."""
    t = text.encode("utf-8")
    return {hash(t[i:i + n]) & 0xFFFFFFFFFFFF for i in range(max(1, len(t) - n + 1))}


def word_shingles(text: str, n: int = 2) -> set[int]:
    """단어 n 개짜리 조각의 집합. 문자 조각보다 우연한 겹침이 훨씬 적다."""
    w = text.split()
    if len(w) < n:
        return {hash(" ".join(w)) & 0xFFFFFFFFFFFF}
    return {hash(" ".join(w[i:i + n])) & 0xFFFFFFFFFFFF for i in range(len(w) - n + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def minhash(shingle_set: set[int], k: int, seed: int = 0) -> np.ndarray:
    """정의 7.10. 해시 k 개 각각의 최소값. (k,) 배열."""
    items = np.fromiter(shingle_set, dtype=np.uint64, count=len(shingle_set))
    salts = np.arange(k, dtype=np.uint64) + np.uint64(seed + 1)
    salts = salts * np.uint64(0x9E3779B97F4A7C15)
    h = mix64(items[:, None] ^ salts[None, :])            # (|S|, k)
    return h.min(axis=0)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """정리 7.11. 서명이 같은 자리의 비율이 자카드의 추정치다."""
    return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """정의 7.13. 서명을 띠 b 개로 나누고 띠마다 해시 테이블을 둔다."""

    def __init__(self, bands: int, rows: int):
        self.b = bands
        self.r = rows
        self.tables: list[dict[int, list[int]]] = [{} for _ in range(bands)]

    def band_keys(self, sig: np.ndarray) -> list[int]:
        keys = []
        for i in range(self.b):
            chunk = sig[i * self.r:(i + 1) * self.r]
            raw = hash(chunk.tobytes()) & 0xFFFFFFFFFFFF
            keys.append(int(mix64(np.array([raw], dtype=np.uint64))[0]))
        return keys

    def insert(self, doc_id: int, sig: np.ndarray) -> None:
        for t, key in zip(self.tables, self.band_keys(sig)):
            t.setdefault(key, []).append(doc_id)

    def candidates(self, sig: np.ndarray) -> set[int]:
        """어느 띠에서든 같은 자리에 온 문서가 후보다."""
        out: set[int] = set()
        for t, key in zip(self.tables, self.band_keys(sig)):
            out.update(t.get(key, []))
        return out


def candidate_probability(j: float, bands: int, rows: int) -> float:
    """명제 7.14. 자카드 j 인 쌍이 적어도 한 띠에서 만날 확률. 1 - (1 - j^r)^b."""
    return 1.0 - (1.0 - j ** rows) ** bands
