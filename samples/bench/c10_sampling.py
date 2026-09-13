"""실험 10.3. 어휘 128,000 에서 뽑기. 누적합 이진 탐색과 별칭표, 그리고 top-p 절단의 값.

지프 분포 확률에서 백만 개를 뽑아 시간을 재고, 뽑은 빈도가 확률과 맞는지 확인한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.sampling import (  # noqa: E402
    AliasTable, sample_cumsum, top_p_mask, top_p_mask_by_select)

V = 128_000
N = 1_000_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    p = 1.0 / np.arange(1, V + 1) ** 1.1
    p /= p.sum()
    rows = []
    ms_build = median_ms(lambda: AliasTable(p), 3)
    table = AliasTable(p)
    ms_cum = median_ms(lambda: sample_cumsum(p, rng, N), 5)
    ms_alias = median_ms(lambda: table.sample(rng, N), 5)
    s1 = sample_cumsum(p, rng, N)
    s2 = table.sample(rng, N)
    f1 = np.bincount(s1, minlength=V)[:10] / N
    f2 = np.bincount(s2, minlength=V)[:10] / N
    err1 = float(np.max(np.abs(f1 - p[:10]) / p[:10]))
    err2 = float(np.max(np.abs(f2 - p[:10]) / p[:10]))
    rows.append(["누적합 + 이진 탐색", 0.0, ms_cum, ms_cum / N * 1e6, err1 * 100])
    rows.append(["별칭표", ms_build, ms_alias, ms_alias / N * 1e6, err2 * 100])
    cols = ["방식", "표 만들기 ms", "백만 개 ms", "뽑기당 ns",
            "상위 10 빈도 오차 퍼센트"]
    cond = (f"어휘 {V:,}, 지프 지수 1.1. 표는 한 번 만들고 백만 개를 뽑는다. "
            "다섯 번 중앙값")
    write_tsv("c10-sampling", cols, rows, "bench/c10_sampling.py", cond, 5)

    rows2 = []
    for top_p in (0.5, 0.9, 0.99):
        m1 = top_p_mask(p, top_p)
        m2 = top_p_mask_by_select(p, top_p, 64)
        assert np.array_equal(m1, m2)
        ms1 = median_ms(lambda: top_p_mask(p, top_p), 5)
        ms2 = median_ms(lambda: top_p_mask_by_select(p, top_p, 64), 5)
        rows2.append([top_p, int(m1.sum()), ms1, ms2])
    cols2 = ["top p", "남는 토큰 수", "전부 정렬 ms", "선택으로 ms"]
    write_tsv("c10-topp", cols2, rows2, "bench/c10_sampling.py",
              "같은 지프 분포. 선택은 상위 64 개부터 두 배씩 늘린다", 5)
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
