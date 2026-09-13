"""실험 12.4. 디스크 위의 그래프 탐색. 페이지 읽기 수.

같은 그래프에서 벡터와 이웃을 한 페이지에 두는 것, 따로 두는 것,
그리고 거친 벡터를 메모리에 두는 것의 페이지 읽기.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.diskgraph import PagedGraph, search_paged, search_with_pq_filter  # noqa: E402
from dsai.nsw import NSW, clustered_points  # noqa: E402

N, D, Q, K, M, EF = 20_000, 64, 100, 10, 16, 64


def main() -> None:
    rng = np.random.default_rng(SEED)
    allp = clustered_points(N + Q, D, 200, rng)
    pts, qs = allp[:N], allp[N:]
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:K].tolist()) for q in qs]
    nbrs = NSW(pts, M).build(rng.permutation(N)).nbrs
    coarse = np.round(pts * 8) / 8          # 거친 벡터. 13장 양자화의 흉내
    rows = []
    layouts = [("벡터와 이웃을 따로", PagedGraph(pts, nbrs, colocated=False)),
               ("한 페이지에 같이", PagedGraph(pts, nbrs, colocated=True))]
    for name, g in layouts:
        fn = search_paged
        reads, rec = [], []
        for q, t in zip(qs, truth):
            res, r = fn(g, q, 0, EF)
            reads.append(r)
            rec.append(len(set(res[:K]) & t) / K)
        rows.append([name, float(np.mean(reads)), float(np.mean(rec)) * 100])
    g = PagedGraph(pts, nbrs, colocated=True)
    reads, rec = [], []
    for q, t in zip(qs, truth):
        res, r = search_with_pq_filter(g, coarse, q, 0, EF)
        reads.append(r)
        rec.append(len(set(res[:K]) & t) / K)
    rows.append(["거친 벡터는 메모리에, 정확한 것만 페이지에서", float(np.mean(reads)),
                 float(np.mean(rec)) * 100])
    cols = ["배치", "질의당 페이지 읽기", "재현율 at 10 퍼센트"]
    cond = (f"군집 200 개의 {D}차원 점 {N:,}개, 이웃 {M}개, ef {EF}, 질의 {Q}개. "
            "페이지는 노드 하나 단위. 거친 벡터는 8 분의 1 단위로 반올림")
    write_tsv("c12-disk", cols, rows, "bench/c12_disk.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
