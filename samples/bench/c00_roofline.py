"""실험 0.2. 이 기계의 루프라인에 연산 셋을 찍는다.

피크 처리량은 큰 행렬 곱으로, 대역폭은 큰 배열 복사로 잰다.
그 위에 내적, 행렬 벡터 곱, 행렬 곱의 밀도와 실측 처리량을 찍는다.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tex, write_tsv  # noqa: E402
from dsai import roofline as rl  # noqa: E402

REPEAT = 7
ELEM = 4  # float32
THREADS = os.cpu_count() or 4


def main() -> None:
    rng = np.random.default_rng(SEED)

    # 대역폭. 코어마다 64MB 를 동시에 복사한다. 코어 하나로는 메모리를 다 못 채운다.
    n_copy = 16 * 1024 * 1024
    srcs = [rng.standard_normal(n_copy, dtype=np.float32) for _ in range(THREADS)]
    dsts = [np.empty_like(s) for s in srcs]

    def copy_all() -> None:
        with ThreadPoolExecutor(THREADS) as pool:
            list(pool.map(lambda i: np.copyto(dsts[i], srcs[i]), range(THREADS)))

    t = median_ms(copy_all, REPEAT)
    bandwidth = 2 * THREADS * n_copy * ELEM / (t / 1000) / 1e9

    # 피크. 2048 x 2048 행렬 곱.
    m = 2048
    a = rng.standard_normal((m, m), dtype=np.float32)
    b = rng.standard_normal((m, m), dtype=np.float32)
    t = median_ms(lambda: rl.matmul(a, b), REPEAT)
    peak = rl.flops_matmul(m, m, m) / (t / 1000) / 1e9

    rows = []
    # 내적. 16M 원소. 캐시보다 훨씬 크다.
    n = 16 * 1024 * 1024
    x = rng.standard_normal(n, dtype=np.float32)
    y = rng.standard_normal(n, dtype=np.float32)
    t = median_ms(lambda: rl.dot(x, y), REPEAT)
    rows.append(["내적", rl.intensity(rl.flops_dot(n), rl.bytes_dot(n, ELEM)),
                 rl.flops_dot(n) / (t / 1000) / 1e9])
    # 행렬 벡터 곱. 4096 x 4096.
    k = 4096
    a2 = rng.standard_normal((k, k), dtype=np.float32)
    v = rng.standard_normal(k, dtype=np.float32)
    t = median_ms(lambda: rl.matvec(a2, v), REPEAT)
    i_mv = rl.intensity(rl.flops_matvec(k, k), rl.bytes_matvec(k, k, ELEM))
    rows.append(["행렬 벡터 곱", i_mv, rl.flops_matvec(k, k) / (t / 1000) / 1e9])
    # 행렬 곱. 위에서 잰 것.
    i_mm = rl.intensity(rl.flops_matmul(m, m, m), rl.bytes_matmul(m, m, m, ELEM))
    rows.append(["행렬 곱", i_mm, peak])

    cond = (f"float32, 밀도는 FLOP/byte, 처리량은 GFLOP/s, "
            f"대역폭은 {THREADS} 스레드 복사")
    write_tsv("c00-roofline", ["연산", "밀도", "처리량"], rows,
              "bench/c00_roofline.py", cond, REPEAT)
    write_tex("c00-roofline-params", {"rlpeak": peak, "rlbw": bandwidth,
                                      "rlridge": rl.ridge_point(peak, bandwidth)})
    print(f"bandwidth {bandwidth:.1f} GB/s, peak {peak:.1f} GFLOP/s")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
