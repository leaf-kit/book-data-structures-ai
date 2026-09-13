"""6장. 해시 테이블. 키를 자리로 바꾸는 함수와,
두 키가 같은 자리에 왔을 때의 세 가지 처리.

체이닝은 자리마다 목록을 두고, 개방 주소는 옆 자리로 밀고,
쿠쿠는 자리 둘 중 하나를 고른다.
셋 다 적재율이 오르면 값이 오르고, 오르는 모양이 다르다.
"""
from __future__ import annotations

import numpy as np

def random_odd(rng: np.random.Generator) -> int:
    """곱셈 자리 이동 해시의 계수. 홀수 64비트 정수 하나."""
    return int(rng.integers(1, 1 << 62)) * 2 + 1


def universal_hash(keys: np.ndarray, a: int, m: int) -> np.ndarray:
    """정의 6.2. 곱셈 자리 이동 해시. (a k mod 2^64) 의 위쪽 l 비트.
    m = 2^l 이어야 한다.

    홀수 a 를 무작위로 고르면 서로 다른 두 키가 같은 자리에 올 확률이 2/m 이하다.
    64비트 넘침이 mod 2^64 노릇을 한다.
    """
    shift = np.uint64(64 - (m.bit_length() - 1))
    return ((keys.astype(np.uint64) * np.uint64(a)) >> shift).astype(np.int64)


def mix64(keys: np.ndarray) -> np.ndarray:
    """비트를 섞는 해시. 곱과 자리 이동을 세 번. 64비트 넘침이 의도된 산수다."""
    z = keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class ChainingTable:
    """정의 6.1 의 체이닝. 자리마다 파이썬 리스트."""

    def __init__(self, m: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.m = m
        self.a = random_odd(rng)
        self.slots: list[list[int]] = [[] for _ in range(m)]

    def slot(self, key: int) -> int:
        return int(universal_hash(np.array([key]), self.a, self.m)[0])

    def insert(self, key: int) -> None:
        self.slots[self.slot(key)].append(key)

    def probes(self, key: int) -> int:
        """찾을 때까지 본 원소 수. 없으면 목록 길이."""
        chain = self.slots[self.slot(key)]
        for i, k in enumerate(chain):
            if k == key:
                return i + 1
        return len(chain)


class LinearProbingTable:
    """정의 6.1 의 개방 주소. 자리가 차 있으면 다음 자리를 본다."""

    EMPTY = -1

    def __init__(self, m: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.m = m
        self.a = random_odd(rng)
        self.table = np.full(m, self.EMPTY, dtype=np.int64)
        self.n = 0

    def slot(self, key: int) -> int:
        return int(universal_hash(np.array([key]), self.a, self.m)[0])

    def insert(self, key: int) -> int:
        """알고리즘 6.1. 비어 있는 자리가 나올 때까지 하나씩 민다.
        본 자리 수를 돌려준다.
        """
        i = self.slot(key)
        probes = 1
        while self.table[i] != self.EMPTY:
            i = (i + 1) % self.m
            probes += 1
        self.table[i] = key
        self.n += 1
        return probes

    def probes(self, key: int) -> int:
        i = self.slot(key)
        p = 1
        while self.table[i] != self.EMPTY:
            if self.table[i] == key:
                return p
            i = (i + 1) % self.m
            p += 1
        return p


class CuckooTable:
    """정의 6.3. 표 둘, 해시 둘. 키는 두 자리 중 하나에 있고 찾기는 최대 두 번 본다."""

    EMPTY = -1

    def __init__(self, m: int, seed: int = 1, max_kicks: int = 500):
        rng = np.random.default_rng(seed)
        self.m = m
        self.h = [random_odd(rng) for _ in range(2)]
        self.tables = [np.full(m, self.EMPTY, dtype=np.int64) for _ in range(2)]
        self.max_kicks = max_kicks
        self.n = 0

    def slot(self, key: int, which: int) -> int:
        return int(universal_hash(np.array([key]), self.h[which], self.m)[0])

    def insert(self, key: int) -> int:
        """자리가 차 있으면 그 키를 쫓아내 상대 표로 보낸다. 쫓아낸 횟수를 돌려준다."""
        which = 0
        for kicks in range(self.max_kicks):
            i = self.slot(key, which)
            if self.tables[which][i] == self.EMPTY:
                self.tables[which][i] = key
                self.n += 1
                return kicks
            key, self.tables[which][i] = int(self.tables[which][i]), key
            which ^= 1
        raise RuntimeError("쫓아내기가 끝나지 않는다. 표를 다시 짜야 한다")

    def probes(self, key: int) -> int:
        if self.tables[0][self.slot(key, 0)] == key:
            return 1
        return 2


def expected_probes_chaining(alpha: float) -> float:
    """정리 6.1. 성공 탐색의 기대 비교 수 1 + alpha/2."""
    return 1.0 + alpha / 2


def expected_probes_linear(alpha: float) -> float:
    """정리 6.1. 선형 탐사 성공 탐색의 기대 자리 수 (1 + 1/(1-alpha)) / 2."""
    return 0.5 * (1.0 + 1.0 / (1.0 - alpha))
