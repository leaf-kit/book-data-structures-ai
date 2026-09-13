"""실험 13.2. 곱 양자화의 바이트 수와 오차, 그리고 비대칭 거리의 재현율.

64차원 float32 (256 바이트) 를 부분 4, 8, 16, 32 개로 양자화하고
재구성 오차와 전부 비교 재현율을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.nsw import clustered_points  # noqa: E402
from dsai.pq import PQ, quantization_error  # noqa: E402

N, D, Q, K = 100_000, 64, 200, 10


def main() -> None:
    rng = np.random.default_rng(SEED)
    allp = clustered_points(N + Q, D, 300, rng)
    pts, qs = allp[:N], allp[N:]
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:K].tolist()) for q in qs]
    rows = [["float32", D * 4, 0.0, 100.0, pts.nbytes / 1e6,
             median_ms(lambda: np.argsort(((pts - qs[0]) ** 2).sum(axis=1))[:K], 5)]]
    for m in [4, 8, 16, 32]:
        pq = PQ(pts[rng.choice(N, 20_000, replace=False)], m, rng, iters=8)
        codes = pq.encode(pts)
        err = quantization_error(pq, pts[:10_000])
        rec = []
        for q, t in zip(qs, truth):
            d = pq.adc(pq.table(q), codes)
            rec.append(len(set(np.argsort(d)[:K].tolist()) & t) / K)
        ms = median_ms(lambda: np.argsort(pq.adc(pq.table(qs[0]), codes))[:K], 5)
        rows.append([f"곱 양자화 m={m}", m, err * 100, float(np.mean(rec)) * 100,
                     codes.nbytes / 1e6, ms])
    cols = ["표현", "바이트", "오차 퍼센트", "재현율 퍼센트", "MB", "질의 ms"]
    cond = (f"군집 300 개의 {D}차원 점 {N:,}개, 질의 {Q}개. "
            "코드북은 표본 2만 개로 학습. "
            "재현율은 코드 전부의 비대칭 거리로 고른 상위 10")
    write_tsv("c13-pq", cols, rows, "bench/c13_pq.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
