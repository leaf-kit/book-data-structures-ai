"""실험 7.2. 카운트 민 스케치로 헤비 히터 찾기.
그리고 하이퍼로그로그로 서로 다른 키 세기.

지프 분포 스트림 천만 개에서 상위 100 개를 스케치로 찾고 정확한 답과 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.sketch import (  # noqa: E402
    CountMinSketch, HyperLogLog, cms_depth_for, cms_width_for)

N = 10_000_000
UNIVERSE = 1_000_000
TOPK = 100


def main() -> None:
    rng = np.random.default_rng(SEED)
    ranks = np.arange(1, UNIVERSE + 1)
    p = 1.0 / ranks ** 1.1
    p /= p.sum()
    ids = rng.choice(UNIVERSE, N, p=p)
    stream = (ids * 7919 + 13).astype(np.int64)          # 키를 흩어 놓는다
    exact = np.bincount(ids, minlength=UNIVERSE)
    top_true = set(np.argsort(-exact)[:TOPK].tolist())
    rows = [["정확한 표", 0, 0, exact.nbytes / 1e6, 100.0, 0.0]]
    for eps, delta in [(1e-3, 1e-2), (1e-4, 1e-2), (1e-5, 1e-3)]:
        w, d = cms_width_for(eps), cms_depth_for(delta)
        cms = CountMinSketch(w, d, seed=1)
        cms.add(stream)
        est_all = cms.estimate((np.arange(UNIVERSE) * 7919 + 13).astype(np.int64))
        top_est = set(np.argsort(-est_all)[:TOPK].tolist())
        recall = len(top_true & top_est) / TOPK
        over = float(np.mean((est_all[list(top_true)] - exact[list(top_true)])
                             / exact[list(top_true)]))
        rows.append([f"스케치 eps={eps:g}", w, d, cms.nbytes() / 1e6,
                     recall * 100, over * 100])
    cols = ["구조", "폭", "깊이", "MB", "상위 100 재현율",
            "과대 추정 퍼센트"]
    cond = (f"지프 분포 스트림 {N:,}개, 서로 다른 키 {UNIVERSE:,}개. "
            "과대 추정은 상위 100 의 평균")
    write_tsv("c07-sketch", cols, rows, "bench/c07_sketch.py", cond, 1)

    rows2 = []
    for p_bits in [8, 12, 16]:
        hll = HyperLogLog(p_bits)
        hll.add(stream)
        true = int(np.count_nonzero(exact))
        est = hll.estimate()
        err = abs(est - true) / true * 100
        rows2.append([1 << p_bits, hll.nbytes() / 1e3, true, est, err])
    cols2 = ["레지스터", "KB", "실제 서로 다른 키", "추정",
             "오차 퍼센트"]
    write_tsv("c07-hll", cols2, rows2, "bench/c07_sketch.py",
              "같은 스트림에서 서로 다른 키의 수", 1)
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
