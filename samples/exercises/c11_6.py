"""문제 11.6. 배열 둘로 둔 명시적 스택의 깊이 우선 위상 정렬."""
import numpy as np

from dsai.graph import CSRGraph, random_dag
from dsai.topo import topo_dfs


def topo_dfs_arrays(g: CSRGraph) -> np.ndarray:
    state = np.zeros(g.n, dtype=np.int8)
    node = np.empty(g.n, dtype=np.int64)          # 스택의 노드
    nxt = np.empty(g.n, dtype=np.int64)           # 스택의 다음 이웃 번호
    order = []
    ptr, col = g.ptr, g.col
    for root in range(g.n):
        if state[root]:
            continue
        top = 0
        node[0], nxt[0] = root, ptr[root]
        state[root] = 1
        while top >= 0:
            u, i = node[top], nxt[top]
            if i < ptr[u + 1]:
                nxt[top] = i + 1
                v = col[i]
                if state[v] == 0:
                    state[v] = 1
                    top += 1
                    node[top], nxt[top] = v, ptr[v]
                elif state[v] == 1:
                    raise ValueError("사이클이 있다")
            else:
                state[u] = 2
                order.append(int(u))
                top -= 1
    return np.array(order[::-1], dtype=np.int64)


def test_array_stack_is_topological():
    rng = np.random.default_rng(1)
    src, dst = random_dag(5000, 20000, rng)
    g = CSRGraph(5000, src, dst)
    order = topo_dfs_arrays(g)
    pos = np.empty(5000, dtype=np.int64)
    pos[order] = np.arange(5000)
    assert np.all(pos[src] < pos[dst])
    assert np.array_equal(order, topo_dfs(g))          # 같은 방문 순서라 같은 답
