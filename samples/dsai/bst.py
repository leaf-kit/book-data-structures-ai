"""8장. 이진 탐색 트리와 균형. 높이가 곧 비용이다.

노드를 배열 넷 (키, 왼쪽, 오른쪽, 우선순위) 으로 둔다. 4장의 인덱스 링크 아레나다.
균형은 트립으로 잡는다. 무작위 우선순위의 힙 조건이 기대 높이를 로그로 묶는다.
"""
from __future__ import annotations

import numpy as np

NIL = -1


class BST:
    """정의 8.1. 순서 불변식만 지키는 트리. 넣는 순서가 높이를 정한다."""

    def __init__(self, capacity: int):
        self.key = np.empty(capacity, dtype=np.int64)
        self.left = np.full(capacity, NIL, dtype=np.int64)
        self.right = np.full(capacity, NIL, dtype=np.int64)
        self.n = 0
        self.root = NIL

    def insert(self, k: int) -> None:
        i = self.n
        self.key[i] = k
        self.n += 1
        if self.root == NIL:
            self.root = i
            return
        cur = self.root
        while True:
            if k < self.key[cur]:
                if self.left[cur] == NIL:
                    self.left[cur] = i
                    return
                cur = self.left[cur]
            else:
                if self.right[cur] == NIL:
                    self.right[cur] = i
                    return
                cur = self.right[cur]

    def search(self, k: int) -> tuple[bool, int]:
        """찾았는지와 들여다본 노드 수. 노드 하나가 라인 하나다."""
        cur, probes = self.root, 0
        while cur != NIL:
            probes += 1
            if k == self.key[cur]:
                return True, probes
            cur = self.left[cur] if k < self.key[cur] else self.right[cur]
        return False, probes

    def height(self) -> int:
        """5장의 명시적 스택으로 잰다. 재귀 한계를 피한다."""
        if self.root == NIL:
            return 0
        best, stack = 0, [(self.root, 1)]
        while stack:
            i, h = stack.pop()
            best = max(best, h)
            for c in (self.left[i], self.right[i]):
                if c != NIL:
                    stack.append((c, h + 1))
        return best


class Treap(BST):
    """정의 8.3. 순서 불변식에 무작위 우선순위의 힙 불변식을 더한다.
    기대 높이 O(log n).
    """

    def __init__(self, capacity: int, seed: int = 0):
        super().__init__(capacity)
        self.prio = np.random.default_rng(seed).random(capacity)

    def _insert_at(self, root: int, i: int) -> int:
        if root == NIL:
            return i
        k = self.key[i]
        if k < self.key[root]:
            self.left[root] = self._insert_at(self.left[root], i)
            if self.prio[self.left[root]] > self.prio[root]:
                root = self._rotate_right(root)
        else:
            self.right[root] = self._insert_at(self.right[root], i)
            if self.prio[self.right[root]] > self.prio[root]:
                root = self._rotate_left(root)
        return root

    def _rotate_right(self, y: int) -> int:
        x = self.left[y]
        self.left[y] = self.right[x]
        self.right[x] = y
        return x

    def _rotate_left(self, x: int) -> int:
        y = self.right[x]
        self.right[x] = self.left[y]
        self.left[y] = x
        return y

    def insert(self, k: int) -> None:
        i = self.n
        self.key[i] = k
        self.n += 1
        self.root = self._insert_at(self.root, i)
