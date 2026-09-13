"""실험 12.1. 차원의 저주. kd-트리가 들여다보는 점의 비율과 거리의 대비.

점 십만 개를 차원 2 에서 128 까지 두고,
최근접 탐색이 들여다본 점의 비율과 최근접 거리 대 평균 거리를 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.kdtree import KDTree, brute_nearest  # noqa: E402

N = 100_000
Q = 100
DIMS = [2, 4, 8, 16, 32, 128]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for d in DIMS:
        pts = rng.standard_normal((N, d)).astype(np.float32)
        tree = KDTree(pts)
        qs = rng.standard_normal((Q, d)).astype(np.float32)
        seen, ok, ratio = [], 0, []
        for q in qs:
            i, s = tree.nearest(q)
            seen.append(s)
            ok += int(i == brute_nearest(pts, q))
            d2 = ((pts - q) ** 2).sum(axis=1)
            ratio.append(float(np.sqrt(d2.min()) / np.sqrt(d2.mean())))
        rows.append([d, float(np.mean(seen)) / N * 100, ok / Q * 100,
                     float(np.mean(ratio))])
    cols = ["차원", "들여다본 점 퍼센트", "정확한 답 퍼센트", "거리 대비"]
    cond = f"표준 정규 점 {N:,}개, 질의 {Q}개. 잎 크기 16 의 kd-트리. 정답은 전부 비교"
    write_tsv("c12-curse", cols, rows, "bench/c12_curse.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
