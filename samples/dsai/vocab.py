"""6장. 어휘 사전과 임베딩 테이블. 토큰을 정수로, 정수를 벡터로.

어휘 사전은 문자열 키의 해시 테이블이다. 만든 뒤에는 바뀌지 않는다.
임베딩 테이블은 정수를 자리로 바로 쓰는 배열이다. 충돌이 없는 해시 테이블이다.
해싱 트릭은 그 배열을 어휘보다 작게 잡고 충돌을 허용한다.
"""
from __future__ import annotations

import bisect
import zlib

import numpy as np


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """자주 나온 순서로 번호를 준다. 번호가 작을수록 흔한 토큰이다."""
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    ordered = sorted(counts, key=lambda t: (-counts[t], t))
    return {t: i for i, t in enumerate(ordered)}


class SortedVocab:
    """정렬된 배열과 이진 탐색. 8장의 디딤돌을 여기서 먼저 쓴다."""

    def __init__(self, vocab: dict[str, int]):
        self.keys = sorted(vocab)
        self.ids = [vocab[k] for k in self.keys]

    def lookup(self, token: str) -> int:
        i = bisect.bisect_left(self.keys, token)
        if i < len(self.keys) and self.keys[i] == token:
            return self.ids[i]
        return -1


def string_hash(token: str, m: int, seed: int = 0) -> int:
    """정의 6.4 의 해싱 트릭. 문자열을 m 개 자리 중 하나로. 사전이 필요 없다."""
    return zlib.crc32(token.encode("utf-8"), seed) % m


class EmbeddingTable:
    """정의 6.4. 번호가 곧 자리인 배열. 조회는 1장의 인덱스 접근이다."""

    def __init__(self, rows: int, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.table = rng.standard_normal((rows, dim), dtype=np.float32) / np.sqrt(dim)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        return self.table[ids]

    def nbytes(self) -> int:
        return self.table.nbytes


def hashed_ids(tokens: list[str], m: int, seed: int = 0) -> np.ndarray:
    return np.array([string_hash(t, m, seed) for t in tokens], dtype=np.int64)


def collision_rate(vocab_size: int, m: int) -> float:
    """명제 6.3.
    어휘 V 개를 자리 m 개에 넣을 때 다른 토큰과 자리를 나누는 토큰의 비율 (기대값).
    """
    if m <= 0:
        return 1.0
    return 1.0 - (1.0 - 1.0 / m) ** (vocab_size - 1)
