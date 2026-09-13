"""12장. 스킵 리스트. 계층으로 건너뛰기.

정렬된 연결 리스트 위에 지름길 층을 확률로 올린다. 노드의 높이가 기하 분포이고,
탐색은 위 층에서 아래로 내려간다.
노드는 4장의 인덱스 링크 아레나이고, 층마다 다음 배열이 하나씩이다.
"""
from __future__ import annotations

import numpy as np

NIL = -1


class SkipList:
    """정의 12.2. 층 L 개의 정렬 리스트.
    층 l 의 노드는 층 l-1 의 노드의 부분 집합이다.
    """

    def __init__(self, capacity: int, max_level: int = 20, p: float = 0.5,
                 seed: int = 0):
        self.key = np.empty(capacity + 1, dtype=np.int64)
        self.nxt = np.full((max_level, capacity + 1), NIL, dtype=np.int64)
        self.level = np.zeros(capacity + 1, dtype=np.int64)
        self.head = 0                      # 노드 0 이 머리. 키는 -무한
        self.key[0] = np.iinfo(np.int64).min
        self.level[0] = max_level
        self.n = 1
        self.top = 0                       # 지금 쓰이는 가장 높은 층
        self.p = p
        self.rng = np.random.default_rng(seed)

    def _random_level(self) -> int:
        h = 1
        while h < self.nxt.shape[0] and self.rng.random() < self.p:
            h += 1
        return h

    def search(self, k: int) -> tuple[bool, int]:
        """알고리즘 12.2. 위 층에서 오른쪽으로 가다 넘치면 내려간다. (찾았는지,
        지나온 노드 수).
        """
        cur, steps = self.head, 0
        for l in range(self.top, -1, -1):
            while self.nxt[l, cur] != NIL and self.key[self.nxt[l, cur]] < k:
                cur = self.nxt[l, cur]
                steps += 1
        cand = self.nxt[0, cur]
        return (cand != NIL and self.key[cand] == k), steps

    def insert(self, k: int) -> None:
        update = [self.head] * self.nxt.shape[0]
        cur = self.head
        for l in range(self.top, -1, -1):
            while self.nxt[l, cur] != NIL and self.key[self.nxt[l, cur]] < k:
                cur = self.nxt[l, cur]
            update[l] = cur
        h = self._random_level()
        if h - 1 > self.top:
            self.top = h - 1
        i = self.n
        self.n += 1
        self.key[i] = k
        self.level[i] = h
        for l in range(h):
            self.nxt[l, i] = self.nxt[l, update[l]]
            self.nxt[l, update[l]] = i

    def nbytes(self) -> int:
        return int(self.key.nbytes + (self.level[: self.n]).sum() * 8)
