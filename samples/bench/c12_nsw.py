"""실험 12.3. 근접 그래프 탐색의 재현율과 거리 계산 수. NSW 와 HNSW, 그리고 전부 비교.

0장의 kNN 문제를 64 차원 점 2만 개에서 다시 푼다.
ef 를 바꾸며 재현율@10 과 거리 계산 수를 잰다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.nsw import HNSW, NSW, clustered_points  # noqa: E402

N, D, Q, K, M = 20_000, 64, 200, 10, 16


def recall(found, truth, k):
    return len(set(found[:k]) & set(truth[:k])) / k


def main() -> None:
    rng = np.random.default_rng(SEED)
    allp = clustered_points(N + Q, D, 200, rng)
    pts, qs = allp[:N], allp[N:]
    truth = [np.argsort(((pts - q) ** 2).sum(axis=1))[:K] for q in qs]
    def brute():
        return np.argpartition(((pts - qs[0]) ** 2).sum(axis=1), K)[:K]
    ms_brute = median_ms(brute, 5)
    order = rng.permutation(N)
    t0 = time.perf_counter()
    nsw = NSW(pts, M).build(order)
    build_nsw = time.perf_counter() - t0
    t0 = time.perf_counter()
    hnsw = HNSW(pts, M, rng, ef_build=32).build(order)
    build_hnsw = time.perf_counter() - t0
    rows = [["전부 비교", 0, 100.0, N, ms_brute]]
    for ef in [10, 32, 64, 128]:
        pair = [("NSW (한 층)", nsw), (f"HNSW ({len(hnsw.layers)} 층)", hnsw)]
        for name, idx in pair:
            rec, nd = [], []
            for q, t in zip(qs, truth):
                res, n = idx.search(q, ef)
                rec.append(recall(res, t, K))
                nd.append(n)
            ms = median_ms(lambda: idx.search(qs[0], ef), 5)
            rows.append([name, ef, float(np.mean(rec)) * 100, float(np.mean(nd)), ms])
    cols = ["구조", "ef", "재현율 at 10 퍼센트", "거리 계산 수", "질의당 ms"]
    cond = (f"군집 200 개의 {D}차원 점 {N:,}개, 질의 {Q}개, 이웃 {M}개. "
            f"만들기 NSW {build_nsw:.0f}초, HNSW {build_hnsw:.0f}초. "
            "시간은 파이썬 루프의 값")
    write_tsv("c12-nsw", cols, rows, "bench/c12_nsw.py", cond, 5)
    for r in rows:
        print(r)
    print(cond)


if __name__ == "__main__":
    main()
