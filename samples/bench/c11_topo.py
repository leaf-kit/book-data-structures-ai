"""실험 11.2. 위상 정렬 둘의 시간, 그리고 스칼라 그래프의 역방향 미분 검산.

노드 20만 개의 DAG 에서 Kahn 과 DFS 를 재고,
무작위 스칼라 그래프의 기울기를 유한 차분과 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.graph import CSRGraph, random_dag  # noqa: E402
from dsai.topo import ScalarGraph, topo_dfs, topo_kahn  # noqa: E402

N = 200_000
M = 800_000


def is_topo(order, src, dst):
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return bool(np.all(pos[src] < pos[dst]))


def random_scalar_graph(rng, n_inputs, n_ops):
    g = ScalarGraph(n_inputs + n_ops)
    for _ in range(n_inputs):
        g.node("input", value=float(rng.uniform(-1, 1)))
    for _ in range(n_ops):
        op = rng.choice(["add", "mul", "tanh"])
        a, b = rng.integers(0, g.n, 2)
        g.node(op, int(a), int(b))
    return g


def main() -> None:
    rng = np.random.default_rng(SEED)
    src, dst = random_dag(N, M, rng)
    g = CSRGraph(N, src, dst)
    ok = topo_kahn(g)
    od = topo_dfs(g)
    assert is_topo(ok, src, dst) and is_topo(od, src, dst)
    rows = [["Kahn (큐와 차수)", N, len(src), median_ms(lambda: topo_kahn(g), 3)],
            ["DFS (명시적 스택)", N, len(src), median_ms(lambda: topo_dfs(g), 3)]]
    cols = ["방법", "노드", "간선", "ms"]
    write_tsv("c11-topo", cols, rows, "bench/c11_topo.py",
              f"노드 {N:,}개, 간선 {len(src):,}개의 DAG. 둘 다 위상 순서임을 확인. "
              "세 번 중앙값", 3)

    sg = random_scalar_graph(rng, 8, 400)
    sg.forward()
    out = sg.n - 1
    grad = sg.backward(out)
    eps = 1e-6
    errs = []
    for i in range(8):
        sg.val[i] += eps
        sg.forward()
        up = sg.val[out]
        sg.val[i] -= 2 * eps
        sg.forward()
        dn = sg.val[out]
        sg.val[i] += eps
        sg.forward()
        fd = (up - dn) / (2 * eps)
        errs.append(abs(fd - grad[i]))
    ms_f = median_ms(sg.forward, 5)
    ms_b = median_ms(lambda: sg.backward(out), 5)
    rows2 = [["전진 한 번", ms_f, 0.0],
             ["역방향 한 번 (모든 입력의 기울기)", ms_b, max(errs)],
             ["유한 차분 (입력 8 개)", ms_f * 16, 0.0]]
    cond2 = ("입력 8 개, 연산 400 개의 무작위 스칼라 그래프. "
             "유한 차분은 입력마다 전진 둘")
    write_tsv("c11-autodiff", ["일", "ms", "유한 차분과의 최대 차이"], rows2,
              "bench/c11_topo.py", cond2, 5)
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
