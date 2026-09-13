"""8장. B-트리. 블록 단위로 탐색하기.

정렬된 키 배열 위에 정적 B-트리를 세운다. 노드 하나가 키 B 개의 연속 블록이고,
탐색은 노드 안에서 이진 탐색, 노드 사이는 자식 링크다.
노드 하나가 라인 하나 (또는 페이지 하나) 이므로 높이가 곧 이동량이다.
"""
from __future__ import annotations

import math

import numpy as np


class StaticBTree:
    """정의 8.6. 차수 B 의 정적 B-트리. 잎이 정렬된 키 전부이고,
    내부 노드는 잎 블록의 첫 키다.
    """

    def __init__(self, keys: np.ndarray, fanout: int):
        assert np.all(keys[:-1] <= keys[1:]), "정렬된 키"
        self.B = fanout
        self.levels: list[np.ndarray] = [keys]        # levels[0] 이 잎
        cur = keys
        while len(cur) > fanout:
            cur = cur[::fanout]                         # 블록마다 첫 키
            self.levels.append(cur)
        self.levels.reverse()                           # levels[0] 이 뿌리

    def height(self) -> int:
        return len(self.levels)

    def search(self, k: int) -> tuple[int, int]:
        """잎에서의 위치 (k 이상인 첫 자리) 와 들여다본 블록 수."""
        pos, blocks = 0, 0
        for lvl in self.levels:
            lo = pos * self.B
            hi = min(lo + self.B, len(lvl))
            blocks += 1
            j = int(np.searchsorted(lvl[lo:hi], k, side="right")) - 1  # 블록 안 탐색
            pos = lo + max(j, 0)
        return pos, blocks

    def nbytes(self) -> int:
        return sum(l.nbytes for l in self.levels[:-1])   # 내부 노드만. 잎은 원래 배열


def expected_height(n: int, fanout: int) -> int:
    """명제 8.7. 키 n 개, 차수 B 의 높이는 ceil(log_B n)."""
    return max(1, math.ceil(math.log(n, fanout)))
