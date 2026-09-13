"""7장. 카운트 민 스케치와 하이퍼로그로그. 개수를 정확히 세지 않고 어림하는 두 표.

카운트 민은 자리 w 개짜리 행 d 개다. 키를 행마다 다른 자리에 더하고,
물을 때 d 값 중 최소를 답한다.
답은 실제보다 크거나 같고, 클 확률은 w 와 d 가 정한다.
"""
from __future__ import annotations

import math

import numpy as np

from dsai.hashing import mix64


class CountMinSketch:
    """정의 7.6. d x w 계수기. 자리는 행마다 다른 해시로 정한다."""

    def __init__(self, width: int, depth: int, seed: int = 0):
        self.w = width
        self.d = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        salts = np.arange(depth, dtype=np.uint64) + np.uint64(seed + 1)
        self.salts = salts * np.uint64(0x9E37)

    def slots(self, keys: np.ndarray) -> np.ndarray:
        h = mix64(keys.astype(np.uint64)[:, None] ^ self.salts[None, :])
        return (h % np.uint64(self.w)).astype(np.int64)      # (n, d)

    def add(self, keys: np.ndarray) -> None:
        """알고리즘 7.1. 행마다 자기 자리를 하나씩 올린다.
        같은 키가 여럿이면 여럿 올린다.
        """
        s = self.slots(keys)
        for r in range(self.d):
            np.add.at(self.table[r], s[:, r], 1)

    def estimate(self, keys: np.ndarray) -> np.ndarray:
        """d 값 중 최소. 남이 더한 것이 섞여 실제 이상이다."""
        s = self.slots(keys)
        rows = np.arange(self.d)
        return self.table[rows[None, :], s].min(axis=1)

    def nbytes(self) -> int:
        return self.table.nbytes


def cms_width_for(eps: float) -> int:
    """정리 7.7. 오차를 전체 개수의 eps 배 안에 두려면 폭 e/eps."""
    return math.ceil(math.e / eps)


def cms_depth_for(delta: float) -> int:
    """그 오차를 넘길 확률을 delta 아래로 두려면 깊이 ln(1/delta)."""
    return math.ceil(math.log(1.0 / delta))


class HyperLogLog:
    """서로 다른 키의 수를 어림한다. 레지스터 m 개에 앞자리 0 의 최대 개수를 적는다."""

    def __init__(self, p: int = 12):
        self.p = p
        self.m = 1 << p
        self.reg = np.zeros(self.m, dtype=np.int8)

    def add(self, keys: np.ndarray) -> None:
        h = mix64(keys.astype(np.uint64))
        idx = (h >> np.uint64(64 - self.p)).astype(np.int64)
        rest = (h << np.uint64(self.p)) | (np.uint64(1) << np.uint64(self.p - 1))
        # 남은 비트에서 첫 1 까지의 0 의 개수 더하기 1
        zeros = np.zeros(len(h), dtype=np.int8)
        r = rest.copy()
        for _ in range(64 - self.p):
            top = (r >> np.uint64(63)) == 0
            zeros += top.astype(np.int8)
            r = np.where(top, r << np.uint64(1), r)
        np.maximum.at(self.reg, idx, zeros + 1)

    def estimate(self) -> float:
        alpha = 0.7213 / (1.0 + 1.079 / self.m)
        raw = alpha * self.m * self.m / np.sum(2.0 ** (-self.reg.astype(np.float64)))
        if raw <= 2.5 * self.m:
            zeros = np.count_nonzero(self.reg == 0)
            if zeros:
                return self.m * math.log(self.m / zeros)
        return float(raw)

    def nbytes(self) -> int:
        return self.reg.nbytes
