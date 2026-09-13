"""12장. 디스크 위의 그래프. 한 번의 접근으로.

메모리에 안 드는 그래프는 노드마다 벡터와 이웃 목록을 한 페이지에 같이 둔다.
탐색이 노드 하나를 넓힐 때 페이지 하나를 읽으면 그 노드의 벡터와 이웃이 함께 온다.
따로 두면 벡터 읽기와 이웃 읽기가 페이지 둘이다. 페이지 읽기 수를 센다.
"""
from __future__ import annotations

import heapq

import numpy as np


class PagedGraph:
    """정의 12.5. 노드마다 (벡터, 이웃 M 개) 를 한 페이지에 둔 배열.
    페이지 읽기를 센다.
    """

    def __init__(self, points: np.ndarray, nbrs, colocated: bool = True):
        self.points = points
        self.nbrs = nbrs
        self.colocated = colocated
        self.page_reads = 0
        self.cache: set[int] = set()

    def _touch(self, page: int) -> None:
        if page not in self.cache:
            self.cache.add(page)
            self.page_reads += 1

    def vector(self, i: int) -> np.ndarray:
        self._touch(i)                                   # 벡터 페이지
        return self.points[i]

    def neighbors(self, i: int):
        self._touch(i if self.colocated else -1 - i)   # 같이 두면 같은 페이지
        return self.nbrs[i]

    def reset(self) -> None:
        self.page_reads = 0
        self.cache = set()


def search_paged(g: PagedGraph, q: np.ndarray, start: int,
                 ef: int) -> tuple[list[int], int]:
    """알고리즘 12.5. 12.3 절의 탐욕 탐색을 페이지 읽기로 센다."""
    g.reset()
    d0 = float(((g.vector(start) - q) ** 2).sum())
    visited = {start}
    cand, best = [(d0, start)], [(-d0, start)]
    while cand:
        d, u = heapq.heappop(cand)
        if d > -best[0][0] and len(best) >= ef:
            break
        for v in g.neighbors(u):
            v = int(v)
            if v in visited:
                continue
            visited.add(v)
            dv = float(((g.vector(v) - q) ** 2).sum())
            if len(best) < ef or dv < -best[0][0]:
                heapq.heappush(cand, (dv, v))
                heapq.heappush(best, (-dv, v))
                if len(best) > ef:
                    heapq.heappop(best)
    return [v for _, v in sorted((-d, v) for d, v in best)], g.page_reads


def d2exact(g: PagedGraph, v: int, q: np.ndarray) -> float:
    return float(((g.vector(v) - q) ** 2).sum())


def search_with_pq_filter(g: PagedGraph, coarse: np.ndarray, q: np.ndarray, start: int,
                          ef: int) -> tuple[list[int], int]:
    """거친 벡터 (coarse) 로 거리를 어림하고, 마지막 후보만 페이지에서 정확히 읽는다."""
    g.reset()
    d0 = float(((coarse[start] - q) ** 2).sum())
    visited = {start}
    cand, best = [(d0, start)], [(-d0, start)]
    while cand:
        d, u = heapq.heappop(cand)
        if d > -best[0][0] and len(best) >= ef:
            break
        for v in g.neighbors(u):                        # 이웃 목록은 페이지에서
            v = int(v)
            if v in visited:
                continue
            visited.add(v)
            dv = float(((coarse[v] - q) ** 2).sum())    # 거리는 메모리의 거친 벡터로
            if len(best) < ef or dv < -best[0][0]:
                heapq.heappush(cand, (dv, v))
                heapq.heappush(best, (-dv, v))
                if len(best) > ef:
                    heapq.heappop(best)
    return sorted((v for _, v in best), key=lambda v: d2exact(g, v, q)), g.page_reads
