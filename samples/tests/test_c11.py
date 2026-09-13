import numpy as np

from dsai.fusion import CHAIN, fusable_chains, fused_blocks, traffic_bytes, unfused
from dsai.graph import AdjList, CSRGraph, random_dag, reach_count_csr, reach_count_list
from dsai.topo import ScalarGraph, topo_dfs, topo_kahn


def small_graph():
    src = np.array([0, 0, 1, 2, 3])
    dst = np.array([1, 2, 3, 3, 4])
    return src, dst


def test_csr_and_list_agree():
    src, dst = small_graph()
    g = CSRGraph(5, src, dst)
    l = AdjList(5)
    for u, v in zip(src, dst):
        l.add_edge(int(u), int(v))
    assert sorted(g.neighbors(0).tolist()) == sorted(l.neighbors(0))
    assert reach_count_csr(g, 0) == reach_count_list(l, 0) == 5
    assert g.transpose().neighbors(3).tolist() == [1, 2]
    rng = np.random.default_rng(0)
    s, d = random_dag(2000, 6000, rng)
    assert np.all(s < d)


def test_topological_orders():
    src, dst = small_graph()
    g = CSRGraph(5, src, dst)
    for order in (topo_kahn(g), topo_dfs(g)):
        pos = np.empty(5, dtype=np.int64)
        pos[order] = np.arange(5)
        assert np.all(pos[src] < pos[dst])
    cyc = CSRGraph(3, np.array([0, 1, 2]), np.array([1, 2, 0]))
    for f in (topo_kahn, topo_dfs):
        try:
            f(cyc)
            assert False
        except ValueError:
            pass


def test_backward_matches_finite_difference():
    g = ScalarGraph(10)
    x = g.node("input", value=0.3)
    y = g.node("input", value=-0.7)
    a = g.node("mul", x, y)
    b = g.node("tanh", a)
    c = g.node("add", b, x)          # x 가 두 번 쓰인다. 팬아웃의 합
    out = g.node("mul", c, c)
    g.forward()
    grad = g.backward(out)
    eps = 1e-6
    for i in (x, y):
        g.val[i] += eps
        g.forward()
        up = g.val[out]
        g.val[i] -= 2 * eps
        g.forward()
        dn = g.val[out]
        g.val[i] += eps
        assert abs((up - dn) / (2 * eps) - grad[i]) < 1e-6


def test_fusion_same_result_and_traffic():
    x = np.random.default_rng(1).standard_normal(100_000)
    out = np.empty_like(x)
    assert np.allclose(fused_blocks(x, out, 4096), unfused(x))
    assert traffic_bytes(10, 5, 8, False) == 5 * traffic_bytes(10, 5, 8, True)
    op = np.array([0, 3, 3, 3, 1, 3, 3])
    indeg = np.array([0, 1, 1, 1, 2, 1, 1])
    outdeg = np.array([1, 1, 1, 1, 1, 1, 0])
    chains = fusable_chains(op, indeg, outdeg, {3})
    assert chains == [[1, 2, 3], [5, 6]]
    assert len(CHAIN) == 5
