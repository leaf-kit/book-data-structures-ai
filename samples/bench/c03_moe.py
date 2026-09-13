"""실험 3.4. 배치 크기에 따라 전문가 혼합이 밀집 층보다 빠른가.

전문가 8 개, 토큰마다 하나를 고른다. 배치가 작으면 전문가 하나만 읽어 빠르고,
배치가 커지면 전문가 전부를 읽게 되어 밀집 층과 같은 이동량이 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.moe import (dense_forward, experts_touched_expected,  # noqa: E402
                      moe_forward, moe_intensity, route_top1)

REPEAT = 7
D, H, E = 2048, 2048, 8
BATCHES = [1, 4, 16, 64, 256]


def main() -> None:
    rng = np.random.default_rng(SEED)
    experts = rng.standard_normal((E, D, H), dtype=np.float32) * 0.02
    dense = rng.standard_normal((D, H), dtype=np.float32) * 0.02   # 전문가 하나 크기
    rows = []
    for b in BATCHES:
        x = rng.standard_normal((b, D), dtype=np.float32)
        assign = route_top1(rng.standard_normal((b, E)))
        t_dense = median_ms(lambda: dense_forward(x, dense), REPEAT)
        t_moe = median_ms(lambda: moe_forward(x, experts, assign), REPEAT)
        touched = experts_touched_expected(E, b)
        rows.append([b, touched, moe_intensity(b, D, H, E, 4), t_dense, t_moe])
    cols = ["배치", "읽는 전문가 수", "밀도", "밀집 ms", "혼합 ms"]
    cond = f"전문가 {E}개, 각 {D}x{H} float32 (16MB). 밀집은 전문가 하나 크기"
    write_tsv("c03-moe", cols, rows, "bench/c03_moe.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
