"""11장. 그래프 표현. 인접 리스트와 CSR 다시 보기.

같은 방향 그래프를 세 가지로 둔다. 리스트의 리스트, 간선 배열 (COO), 그리고 3장의 CSR.
계산 그래프는 노드가 연산이고 간선이 텐서다. 정적이고 읽기가 대부분이라 CSR 이 맞다.
"""
from __future__ import annotations

import numpy as np


class AdjList:
    """정의 11.1 의 인접 리스트. 노드마다 파이썬 리스트. 노드가 흩어져 있다."""

    def __init__(self, n: int):
        self.n = n
        self.out: list[list[int]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int) -> None:
        self.out[u].append(v)

    def neighbors(self, u: int) -> list[int]:
        return self.out[u]

    def nbytes(self) -> int:
        import sys
        return sum(sys.getsizeof(l) + 28 * len(l) for l in self.out) + 8 * self.n


class CSRGraph:
    """정의 11.1 의 CSR. 행 포인터 하나와 열 배열 하나. 3장의 CSR 을 그래프로 읽는다."""

    def __init__(self, n: int, src: np.ndarray, dst: np.ndarray):
        order = np.argsort(src, kind="stable")
        self.n = n
        self.col = dst[order].astype(np.int64)
        counts = np.bincount(src, minlength=n)
        self.ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=self.ptr[1:])

    def neighbors(self, u: int) -> np.ndarray:
        return self.col[self.ptr[u]:self.ptr[u + 1]]

    def out_degree(self) -> np.ndarray:
        return np.diff(self.ptr)

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.col, minlength=self.n)

    def nbytes(self) -> int:
        return self.ptr.nbytes + self.col.nbytes

    def transpose(self) -> "CSRGraph":
        """들어오는 간선의 CSR. 역방향 미분이 이것을 따라간다."""
        src = np.repeat(np.arange(self.n), np.diff(self.ptr))
        return CSRGraph(self.n, self.col, src)


def random_dag(n: int, m: int, rng: np.random.Generator,
               p: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """노드 번호가 위상 순서인 무작위 DAG. 간선은 작은 번호에서 큰 번호로.
    간격은 기하 분포 (평균 1/p).
    """
    src = rng.integers(0, n - 1, m)
    span = rng.geometric(p, m)                        # 대부분 가까운 노드로 간다
    dst = np.minimum(src + span, n - 1)
    keep = dst > src
    return src[keep], dst[keep]


def degree_sum_list(g: AdjList) -> int:
    """디딤돌. 모든 노드의 이웃을 한 번씩 훑는다. 노드마다 리스트 하나를 따라간다."""
    return sum(len(g.neighbors(u)) for u in range(g.n))


def degree_sum_csr(g: CSRGraph) -> int:
    """같은 훑기. 포인터 배열의 차이를 더한다. 열 배열은 안 읽는다."""
    return int(g.out_degree().sum())


def reach_count_list(g: AdjList, start: int) -> int:
    """디딤돌. 너비 우선 탐색으로 닿는 노드 수. 5장의 큐."""
    seen = bytearray(g.n)
    seen[start] = 1
    frontier = [start]
    count = 1
    while frontier:
        nxt = []
        for u in frontier:
            for v in g.neighbors(u):
                if not seen[v]:
                    seen[v] = 1
                    count += 1
                    nxt.append(v)
        frontier = nxt
    return count


def reach_count_csr(g: CSRGraph, start: int) -> int:
    """알고리즘 11.1. 같은 탐색을 CSR 위에서 층마다 배열 연산으로."""
    seen = np.zeros(g.n, dtype=bool)
    seen[start] = True
    frontier = np.array([start])
    count = 1
    while len(frontier):
        starts, ends = g.ptr[frontier], g.ptr[frontier + 1]
        idx = _ranges(starts, ends)
        cand = np.unique(g.col[idx])
        cand = cand[~seen[cand]]
        seen[cand] = True
        count += len(cand)
        frontier = cand
    return count


def _ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """[s_i, e_i) 구간들을 이어 붙인 번호 배열. 반복문 없이."""
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return np.arange(total) + offs
