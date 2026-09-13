"""11장. 위상 정렬과 자동 미분. 계산 그래프를 앞으로 한 번, 뒤로 한 번.

Kahn 의 위상 정렬은 들어오는 간선 수를 세고 0 인 노드를 큐로 뺀다.
DFS 는 5장의 명시적 스택이다.
역방향 미분은 위상 순서의 역순으로 노드를 지나며, 나가는 간선마다 온 기울기를 더한다.
5장의 테이프가 선이었다면 여기는 DAG 다.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from dsai.graph import CSRGraph


def topo_kahn(g: CSRGraph) -> np.ndarray:
    """알고리즘 11.2. 들어오는 간선이 0 인 노드부터. 큐 하나와 차수 배열 하나."""
    indeg = g.in_degree().copy()
    q = deque(np.flatnonzero(indeg == 0).tolist())
    order = np.empty(g.n, dtype=np.int64)
    k = 0
    while q:
        u = q.popleft()
        order[k] = u
        k += 1
        for v in g.neighbors(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if k != g.n:
        raise ValueError("사이클이 있다")
    return order


def topo_dfs(g: CSRGraph) -> np.ndarray:
    """알고리즘 11.3. 깊이 우선 탐색의 끝난 순서를 뒤집는다. 명시적 스택."""
    state = np.zeros(g.n, dtype=np.int8)          # 0 안 봄, 1 보는 중, 2 끝
    order = []
    for root in range(g.n):
        if state[root]:
            continue
        stack = [(root, 0)]
        state[root] = 1
        while stack:
            u, i = stack[-1]
            nb = g.neighbors(u)
            if i < len(nb):
                stack[-1] = (u, i + 1)
                v = int(nb[i])
                if state[v] == 0:
                    state[v] = 1
                    stack.append((v, 0))
                elif state[v] == 1:
                    raise ValueError("사이클이 있다")
            else:
                state[u] = 2
                order.append(u)
                stack.pop()
    return np.array(order[::-1], dtype=np.int64)


class ScalarGraph:
    """정의 11.5. 스칼라 계산 그래프. 노드는 연산이고 부모 둘까지.
    값과 기울기를 배열에 둔다.
    """

    OPS = {"input": 0, "add": 1, "mul": 2, "tanh": 3}

    def __init__(self, capacity: int):
        self.op = np.zeros(capacity, dtype=np.int8)
        self.a = np.full(capacity, -1, dtype=np.int64)
        self.b = np.full(capacity, -1, dtype=np.int64)
        self.val = np.zeros(capacity)
        self.n = 0

    def node(self, op: str, a: int = -1, b: int = -1, value: float = 0.0) -> int:
        i = self.n
        self.op[i], self.a[i], self.b[i], self.val[i] = self.OPS[op], a, b, value
        self.n += 1
        return i

    def forward(self) -> None:
        """노드 번호가 위상 순서다. 앞에서 뒤로 한 번."""
        for i in range(self.n):
            o = self.op[i]
            if o == 1:
                self.val[i] = self.val[self.a[i]] + self.val[self.b[i]]
            elif o == 2:
                self.val[i] = self.val[self.a[i]] * self.val[self.b[i]]
            elif o == 3:
                self.val[i] = np.tanh(self.val[self.a[i]])

    def backward(self, out: int) -> np.ndarray:
        """알고리즘 11.4. 뒤에서 앞으로 한 번.
        부모마다 기울기를 더한다 (팬아웃의 합).
        """
        grad = np.zeros(self.n)
        grad[out] = 1.0
        for i in range(out, -1, -1):
            o, g = self.op[i], grad[i]
            if g == 0.0:
                continue
            if o == 1:
                grad[self.a[i]] += g
                grad[self.b[i]] += g
            elif o == 2:
                grad[self.a[i]] += g * self.val[self.b[i]]
                grad[self.b[i]] += g * self.val[self.a[i]]
            elif o == 3:
                grad[self.a[i]] += g * (1.0 - self.val[i] ** 2)
        return grad
