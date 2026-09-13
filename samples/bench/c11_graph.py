"""실험 11.1. 같은 DAG, 세 표현. 메모리와 너비 우선 탐색 시간.

노드 백만 개,
간선 사백만 개의 무작위 DAG 를 리스트의 리스트와 CSR 로 두고 닿는 노드 수를 센다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.graph import (  # noqa: E402
    AdjList, CSRGraph, degree_sum_csr, degree_sum_list, random_dag, reach_count_csr,
    reach_count_list)

N = 1_000_000
M = 4_000_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    src, dst = random_dag(N, M, rng, p=0.002)
    ms_build_list = median_ms(lambda: build_list(src, dst), 1)
    g_list = build_list(src, dst)
    ms_build_csr = median_ms(lambda: CSRGraph(N, src, dst), 3)
    g_csr = CSRGraph(N, src, dst)
    a = reach_count_list(g_list, 0)
    b = reach_count_csr(g_csr, 0)
    assert a == b
    ms_list = median_ms(lambda: reach_count_list(g_list, 0), 3)
    ms_csr = median_ms(lambda: reach_count_csr(g_csr, 0), 3)
    assert degree_sum_list(g_list) == degree_sum_csr(g_csr) == len(src)
    ms_dl = median_ms(lambda: degree_sum_list(g_list), 3)
    ms_dc = median_ms(lambda: degree_sum_csr(g_csr), 5)
    rows = [["리스트의 리스트", g_list.nbytes() / 1e6, ms_build_list, ms_dl, ms_list],
            ["CSR (포인터 + 열)", g_csr.nbytes() / 1e6, ms_build_csr, ms_dc, ms_csr],
            ["간선 배열 (COO)", (src.nbytes + dst.nbytes) / 1e6, 0.0, 0.0, 0.0]]
    cols = ["표현", "MB", "만들기 ms", "전체 훑기 ms", "너비 우선 탐색 ms"]
    cond = (f"노드 {N:,}개, 간선 {len(src):,}개, 간선 간격 평균 500. "
            f"탐색은 노드 0 에서 닿는 {a:,}개, 층 단위. 세 번 중앙값")
    write_tsv("c11-graph", cols, rows, "bench/c11_graph.py", cond, 3)
    for r in rows:
        print(r)


def build_list(src, dst):
    g = AdjList(N)
    for u, v in zip(src.tolist(), dst.tolist()):
        g.add_edge(u, v)
    return g


if __name__ == "__main__":
    main()
