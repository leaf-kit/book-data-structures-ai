"""실험 13.4. 인덱스 고르기. 같은 점, 같은 질의,
같은 재현율 목표에서 네 구조의 메모리와 지연.

전부 비교, 역파일, 곱 양자화, 역파일 + 곱 양자화, 근접 그래프.
재현율 90 퍼센트 이상이 되는 가장 싼 설정을 적는다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.ivf import IVF, kmeans  # noqa: E402
from dsai.ivfpq import IVFPQ  # noqa: E402
from dsai.nsw import NSW, clustered_points  # noqa: E402
from dsai.select import recall_curve  # noqa: E402
from dsai.pq import PQ  # noqa: E402

N, D, Q, K, C, M, TARGET = 50_000, 64, 100, 10, 512, 8, 0.9


def pick(fn, params, qs, truth):
    """recall_curve 로 설정을 고르고 그 설정의 질의당 ms 를 잰다. 미달이면 마지막."""
    p, rec = recall_curve(fn, params, qs, truth, K, TARGET)
    p = params[-1] if p is None else p
    return p, rec * 100, median_ms(lambda: fn(qs[0], p), 5)


def main() -> None:
    rng = np.random.default_rng(SEED)
    allp = clustered_points(N + Q, D, 200, rng)
    pts, qs = allp[:N], allp[N:]
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:K].tolist()) for q in qs]
    sample = pts[rng.choice(N, 20_000, replace=False)]
    rows = []
    ms = median_ms(lambda: np.argpartition(((pts - qs[0]) ** 2).sum(axis=1), K)[:K], 5)
    rows.append(["전부 비교", "없음", 100.0, pts.nbytes / 1e6, 0.0, ms])

    t0 = time.perf_counter()
    cent = kmeans(sample, C, rng, iters=10)
    ivf = IVF(pts, cent)
    build = time.perf_counter() - t0
    p, rec, ms = pick(lambda q, np_: ivf.search(q, np_, K)[0],
                              [1, 2, 4, 8, 16, 32, 64], qs, truth)
    mb = (ivf.xs.nbytes + ivf.ids.nbytes) / 1e6
    rows.append(["역파일", f"nprobe {p}", rec, mb, build, ms])

    t0 = time.perf_counter()
    pq = PQ(sample, M, rng, iters=8)
    codes = pq.encode(pts)
    build = time.perf_counter() - t0
    p, rec, ms = pick(lambda q, _: np.argsort(pq.adc(pq.table(q), codes))[:K],
                              [0], qs, truth)
    rows.append(["곱 양자화", f"부분 {M}", rec, codes.nbytes / 1e6, build, ms])

    t0 = time.perf_counter()
    ivfpq = IVFPQ(pts, cent, M, rng)
    build = time.perf_counter() - t0
    p, rec, ms = pick(lambda q, np_: ivfpq.search(q, np_, K, 100)[0],
                              [4, 8, 16, 32, 64, 128], qs, truth)
    rows.append(["역파일 + PQ + 다시 재기", f"nprobe {p}", rec,
                 ivfpq.nbytes() / 1e6, build, ms])

    t0 = time.perf_counter()
    g = NSW(pts, 16, 32).build(rng.permutation(N))
    build = time.perf_counter() - t0
    p, rec, ms = pick(lambda q, ef: g.search(q, ef)[0][:K],
                              [10, 20, 40, 80, 160], qs, truth)
    mb = (pts.nbytes + g.edges() * 8) / 1e6
    rows.append(["근접 그래프", f"ef {p}", rec, mb, build, ms])
    cols = ["구조", "설정", "재현율 퍼센트", "MB", "만들기 초", "질의 ms"]
    cond = (f"군집 200 개의 {D}차원 점 {N:,}개, 질의 {Q}개, "
            f"목표 재현율 {TARGET * 100:.0f} 퍼센트. "
            "시간은 파이썬과 NumPy 의 값이라 비율만 읽는다")
    write_tsv("c13-select", cols, rows, "bench/c13_select.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
