"""실험 16.1. 별과 링. 기계 수를 올리며 기계 하나가 보내는 바이트와 모형 시간.

벡터 100만 개 (8 MB) 를 기계 p 대가 합친다.
별은 뿌리 하나의 링크가, 링은 모든 링크가 고르게 든다.
모형 시간은 대역폭 10 GB 매초, 단계마다 지연 20 마이크로초로 셈한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.collective import (allreduce_ring, allreduce_star,  # noqa: E402
                             bytes_per_machine)

N = 1_000_000
BW = 10e9          # 바이트 매초
LAT = 20e-6        # 단계당 초


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for p in [2, 4, 8, 16, 64]:
        shards = [rng.standard_normal(N) for _ in range(p)]
        t_star, b_star, s_star = allreduce_star(shards)
        t_ring, b_ring, s_ring = allreduce_ring(shards)
        assert np.allclose(t_star, t_ring)
        assert b_star == bytes_per_machine(N, 8, p, "star")
        ms_star = (b_star / BW + s_star * LAT) * 1e3
        ms_ring = (b_ring / BW + s_ring * LAT) * 1e3
        rows.append([p, b_star / 1e6, b_ring / 1e6, s_star, s_ring, ms_star, ms_ring])
    cols = ["기계 p", "별 뿌리 MB", "링 링크 MB", "별 단계", "링 단계", "별 모형 ms",
            "링 모형 ms"]
    cond = (f"벡터 {N:,} 개 (float64, 8 MB). 바이트는 시뮬레이션에서 실제로 센 값. "
            "모형 시간은 대역폭 10 GB 매초, 단계당 지연 20 마이크로초. "
            "두 방법의 합이 같음을 확인")
    write_tsv("c16-allreduce", cols, rows, "bench/c16_allreduce.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
