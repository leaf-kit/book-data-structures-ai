"""12장. kd-트리와 차원의 저주. 낮은 차원에서 로그였던 탐색이 높은 차원에서 선형이 된다.

축 하나씩 번갈아 중앙값으로 나누는 정적 kd-트리.
최근접 탐색은 가지치기로 부분 트리를 건너뛴다.
차원이 오르면 가지치기가 안 되고 거의 전부를 본다. 그것을 들여다본 점의 비율로 잰다.
"""
from __future__ import annotations

import numpy as np


class KDTree:
    """정의 12.1. 축을 번갈아 중앙값으로 나눈 이진 트리. 잎이 점 몇 개의 묶음이다."""

    def __init__(self, points: np.ndarray, leaf_size: int = 16):
        self.pts = points
        n, d = points.shape
        self.d = d
        self.leaf_size = leaf_size
        self.idx = np.arange(n)
        # 노드는 (축, 분할값, 왼쪽, 오른쪽) 이고 잎은 (-1, lo, hi, 0)
        self.nodes: list[tuple] = []
        self.root = self._build(0, n, 0)

    def _build(self, lo: int, hi: int, depth: int) -> int:
        if hi - lo <= self.leaf_size:
            self.nodes.append((-1, lo, hi, 0))
            return len(self.nodes) - 1
        axis = depth % self.d
        sub = self.idx[lo:hi]
        order = np.argsort(self.pts[sub, axis], kind="stable")
        self.idx[lo:hi] = sub[order]
        mid = (lo + hi) // 2
        split = float(self.pts[self.idx[mid], axis])
        me = len(self.nodes)
        self.nodes.append(None)
        left = self._build(lo, mid, depth + 1)
        right = self._build(mid, hi, depth + 1)
        self.nodes[me] = (axis, split, left, right)
        return me

    def nearest(self, q: np.ndarray) -> tuple[int, int]:
        """알고리즘 12.1. 가지치기 최근접 탐색. (가장 가까운 점의 번호,
        들여다본 점 수).
        """
        best, best_d2, seen = -1, np.inf, 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            axis, a, b, c = self.nodes[node]
            if axis == -1:
                ids = self.idx[a:b]
                d2 = ((self.pts[ids] - q) ** 2).sum(axis=1)
                seen += len(ids)
                j = int(np.argmin(d2))
                if d2[j] < best_d2:
                    best, best_d2 = int(ids[j]), float(d2[j])
                continue
            diff = q[axis] - a
            near, far = (b, c) if diff < 0 else (c, b)
            if diff * diff < best_d2:             # 먼 쪽이 아직 이길 수 있을 때만 본다
                stack.append(far)
            stack.append(near)
        return best, seen


def brute_nearest(points: np.ndarray, q: np.ndarray) -> int:
    return int(np.argmin(((points - q) ** 2).sum(axis=1)))
