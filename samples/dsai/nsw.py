"""12장. 탐색 가능한 작은 세상 그래프. NSW 와 HNSW.

점을 하나씩 넣는다.
새 점은 지금까지의 그래프에서 탐욕 탐색으로 찾은 가까운 점 M 개와 이어진다.
먼저 들어온 점은 점이 드물 때 이어졌으므로 긴 간선을 갖고,
그것이 작은 세상의 지름길이다.
HNSW 는 스킵 리스트처럼 층을 올린다. 위 층은 점이 드물어 멀리 건너뛰고,
아래 층에서 정확히 찾는다.
"""
from __future__ import annotations

import heapq

import numpy as np


def d2(points: np.ndarray, i: int, q: np.ndarray) -> float:
    return float(((points[i] - q) ** 2).sum())


def greedy_search(points: np.ndarray, nbrs: list[list[int]], q: np.ndarray, start: int,
                  ef: int) -> tuple[list[tuple[float, int]], int]:
    """알고리즘 12.3. 후보 ef 개를 유지하는 탐욕 탐색. (가까운 순, 계산 수)."""
    d0 = d2(points, start, q)
    visited = {start}
    cand = [(d0, start)]                     # 최소 힙. 아직 안 넓힌 후보
    best = [(-d0, start)]                    # 최대 힙. 지금까지의 상위 ef
    n_dist = 1
    while cand:
        d, u = heapq.heappop(cand)
        if d > -best[0][0] and len(best) >= ef:
            break                    # 가장 가까운 후보가 결과의 최악보다 멀면 끝
        for v in nbrs[u]:
            if v in visited:
                continue
            visited.add(v)
            dv = d2(points, v, q)
            n_dist += 1
            if len(best) < ef or dv < -best[0][0]:
                heapq.heappush(cand, (dv, v))
                heapq.heappush(best, (-dv, v))
                if len(best) > ef:
                    heapq.heappop(best)
    return sorted((-d, v) for d, v in best), n_dist


class NSW:
    """정의 12.3. 삽입 순서로 만드는 근접 그래프. 새 점은 M 개와 잇고,
    목록은 2M 까지 자란다.
    """

    def __init__(self, points: np.ndarray, m: int, ef_build: int = 32,
                 cap: int | None = None):
        self.points = points
        self.m = m
        self.cap = cap or 2 * m
        self.ef_build = ef_build
        self.nbrs: list[list[int]] = [[] for _ in range(len(points))]
        self.entry = -1
        self.size = 0

    def select(self, v: int, cand: list[int], m: int) -> list[int]:
        """이웃 고르기. 가까운 순으로 보되, 이미 고른 이웃 하나가 후보보다 v 에 가깝고
        후보에도 더 가까우면 (후보가 그 이웃 너머에 있으면) 버린다."""
        pv = self.points[v]
        cand = sorted(cand, key=lambda w: d2(self.points, w, pv))
        kept: list[int] = []
        for w in cand:
            dvw = d2(self.points, w, pv)
            if all(d2(self.points, u, self.points[w]) >= dvw for u in kept):
                kept.append(w)
                if len(kept) >= m:
                    break
        return kept

    def insert(self, i: int) -> None:
        """알고리즘 12.4. 탐욕 탐색으로 후보를 찾아 M 개를 고르고 양방향으로 잇는다.
        넘치면 다시 고른다.
        """
        if self.entry < 0:
            self.entry = i
            self.size = 1
            return
        found, _ = greedy_search(self.points, self.nbrs, self.points[i], self.entry,
                                 self.ef_build)
        self.nbrs[i] = self.select(i, [v for _, v in found], self.m)
        for v in self.nbrs[i]:
            self.nbrs[v].append(i)
            if len(self.nbrs[v]) > self.cap:
                self.nbrs[v] = self.select(v, self.nbrs[v], self.cap)
        self.size += 1

    def build(self, order: np.ndarray) -> "NSW":
        for i in order:
            self.insert(int(i))
        return self

    def search(self, q: np.ndarray, ef: int) -> tuple[list[int], int]:
        found, n = greedy_search(self.points, self.nbrs, q, self.entry, ef)
        return [v for _, v in found], n

    def edges(self) -> int:
        return sum(len(l) for l in self.nbrs)


class HNSW:
    """정의 12.4. 층마다 NSW. 점의 층 높이는 기하 분포이고,
    위 층부터 ef 1 로 내려온 뒤 층 0 에서 ef 로 넓힌다.
    """

    def __init__(self, points: np.ndarray, m: int, rng: np.random.Generator,
                 p: float = 0.25, ef_build: int = 32):
        self.points = points
        self.levels = np.minimum(rng.geometric(1 - p, len(points)) - 1, 5)
        top = int(self.levels.max())
        self.layers = [NSW(points, m, ef_build) for _ in range(top + 1)]

    def build(self, order: np.ndarray) -> "HNSW":
        for i in order:
            for l in range(int(self.levels[i]) + 1):
                self.layers[l].insert(int(i))
        return self

    def search(self, q: np.ndarray, ef: int) -> tuple[list[int], int]:
        """알고리즘 12.5.
        맨 위 층의 진입점에서 층을 내려오며 가장 가까운 점 하나를 따라간다.
        """
        cur, total = self.layers[-1].entry, 0
        for l in range(len(self.layers) - 1, 0, -1):
            found, n = greedy_search(self.points, self.layers[l].nbrs, q, cur, 1)
            cur, total = found[0][1], total + n
        found, n = greedy_search(self.points, self.layers[0].nbrs, q, cur, ef)
        return [v for _, v in found], total + n


def clustered_points(n: int, d: int, clusters: int,
                     rng: np.random.Generator) -> np.ndarray:
    """군집이 있는 점. 임베딩처럼 뭉쳐 있다."""
    centers = rng.standard_normal((clusters, d)).astype(np.float32)
    labels = rng.integers(0, clusters, n)
    pts = centers[labels] + 0.3 * rng.standard_normal((n, d)).astype(np.float32)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)
