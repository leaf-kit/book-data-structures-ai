"""실험 13.1. 역파일. 군집 수와 nprobe 에 따른 재현율과 견준 점 수.

64차원 점 10만 개를 군집 1,024 개로 나누고,
nprobe 를 바꾸며 재현율@10 과 견준 점의 비율을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.ivf import IVF, kmeans  # noqa: E402
from dsai.nsw import clustered_points  # noqa: E402

N, D, Q, K, C = 100_000, 64, 200, 10, 1024


def main() -> None:
    rng = np.random.default_rng(SEED)
    allp = clustered_points(N + Q, D, 300, rng)
    pts, qs = allp[:N], allp[N:]
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:K].tolist()) for q in qs]
    def brute():
        return np.argpartition(((pts - qs[0]) ** 2).sum(axis=1), K)[:K]
    ms_brute = median_ms(brute, 5)
    cent = kmeans(pts[rng.choice(N, 20_000, replace=False)], C, rng, iters=10)
    ivf = IVF(pts, cent)
    rows = [["전부 비교", 0, 100.0, 100.0, ms_brute]]
    for nprobe in [1, 4, 16, 64]:
        rec, seen = [], []
        for q, t in zip(qs, truth):
            res, n = ivf.search(q, nprobe, K)
            rec.append(len(set(res.tolist()) & t) / K)
            seen.append(n)
        ms = median_ms(lambda: ivf.search(qs[0], nprobe, K), 5)
        rows.append([f"역파일 {C}", nprobe, float(np.mean(rec)) * 100,
                     float(np.mean(seen)) / N * 100, ms])
    cols = ["구조", "nprobe", "재현율 at 10 퍼센트", "견준 점 퍼센트", "질의당 ms"]
    cond = (f"군집 300 개의 {D}차원 점 {N:,}개, 질의 {Q}개. "
            "k-평균은 표본 2만 개로 10 번 반복")
    write_tsv("c13-ivf", cols, rows, "bench/c13_ivf.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
