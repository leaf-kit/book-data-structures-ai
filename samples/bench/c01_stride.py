"""실험 1.3. 간격을 바꿔 가며 훑기. 원소당 시간이 라인 폭에서 꺾인다.

같은 배열을 k 개마다 하나씩 읽는다.
읽는 원소 수는 줄지만, 원소당 시간은 k 가 16 이 될 때까지 오른다.
float32 열여섯 개가 라인 하나이기 때문이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.tensor import elements_per_line  # noqa: E402
from dsai.tiling import strided_sum  # noqa: E402

REPEAT = 7
N = 64 * 1024 * 1024  # 256MB
LINE = 64
STEPS = [1, 2, 4, 8, 16, 32, 64, 128]


def main() -> None:
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal(N, dtype=np.float32)
    rows = []
    for k in STEPS:
        t = median_ms(lambda: strided_sum(a, k), REPEAT)
        touched = (N + k - 1) // k
        rows.append([k, touched, t, t * 1e6 / touched, elements_per_line(4 * k, LINE)])
    cols = ["간격", "읽은 원소", "시간 ms", "원소당 ns", "라인당 원소"]
    cond = "float32 6,400만 개 (256MB), k 개마다 하나씩 읽어 더함"
    write_tsv("c01-stride", cols, rows, "bench/c01_stride.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
