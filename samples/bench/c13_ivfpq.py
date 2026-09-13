"""실험 13.3. 역파일 + 곱 양자화. 잔차 양자화의 오차와, 다시 재기의 효과.

같은 점 10만 개에서 군집 1,024,
부분 8 의 IVFPQ 를 세우고 nprobe 와 다시 재기 수를 바꾼다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.ivf import kmeans  # noqa: E402
from dsai.ivfpq import IVFPQ  # noqa: E402
from dsai.nsw import clustered_points  # noqa: E402
from dsai.pq import PQ, quantization_error  # noqa: E402

N, D, Q, K, C, M = 100_000, 64, 200, 10, 1024, 8


def main() -> None:
    rng = np.random.default_rng(SEED)
    allp = clustered_points(N + Q, D, 300, rng)
    pts, qs = allp[:N], allp[N:]
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:K].tolist()) for q in qs]
    sample = pts[rng.choice(N, 20_000, replace=False)]
    cent = kmeans(sample, C, rng, iters=10)
    idx = IVFPQ(pts, cent, M, rng)
    pq_plain = PQ(sample, M, rng, iters=8)
    err_plain = quantization_error(pq_plain, pts[:10_000])
    a = np.repeat(np.arange(C), np.diff(idx.ivf.ptr))
    resid = idx.ivf.xs[:10_000] - cent[a[:10_000]]
    rec_resid = idx.pq.decode(idx.pq.encode(resid))
    err_resid = float(((resid - rec_resid) ** 2).sum(axis=1).mean()
                      / (pts[:10_000] ** 2).sum(axis=1).mean())
    rows = []
    for nprobe, rerank in [(8, 0), (32, 0), (32, 100), (128, 0), (128, 100)]:
        rec, seen = [], []
        for q, t in zip(qs, truth):
            res, n = idx.search(q, nprobe, K, rerank)
            rec.append(len(set(res.tolist()) & t) / K)
            seen.append(n)
        ms = median_ms(lambda: idx.search(qs[0], nprobe, K, rerank), 5)
        rows.append([nprobe, rerank, float(np.mean(rec)) * 100,
                     float(np.mean(seen)) / N * 100, ms])
    cols = ["nprobe", "다시 잰 후보", "재현율 at 10 퍼센트", "견준 점 퍼센트",
            "질의당 ms"]
    cond = (f"{D}차원 점 {N:,}개, 군집 {C}, 부분 {M} (벡터당 {M} 바이트). "
            f"인덱스 {idx.nbytes() / 1e6:.1f} MB. 재구성 오차는 원 벡터 "
            f"{err_plain * 100:.1f} 퍼센트, 잔차 {err_resid * 100:.1f} 퍼센트")
    write_tsv("c13-ivfpq", cols, rows, "bench/c13_ivfpq.py", cond, 5)
    for r in rows:
        print(r)
    print(cond)


if __name__ == "__main__":
    main()
