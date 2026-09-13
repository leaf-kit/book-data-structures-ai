"""실험 3.1. 밀도에 따라 밀집 곱과 CSR 곱 중 어느 쪽이 빠른가.

같은 4096 x 4096 행렬에서 0 이 아닌 비율을 바꿔 가며 행렬 벡터 곱을 잰다.
어느 밀도부터 밀집이 이기는지가 이 실험의 답이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.sparse import csr_matvec, dense_to_csr, random_sparse  # noqa: E402

REPEAT = 7
N = 4096
DENSITIES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N, dtype=np.float32)
    rows = []
    for d in DENSITIES:
        a = random_sparse(N, N, d, SEED)
        m = dense_to_csr(a)
        t_dense = median_ms(lambda: a @ x, REPEAT)
        t_csr = median_ms(lambda: csr_matvec(m, x), REPEAT)
        rows.append([d * 100, m.nnz, a.nbytes / 1e6, m.nbytes() / 1e6, t_dense, t_csr])
    cols = ["밀도 퍼센트", "nnz", "밀집 MB", "CSR MB", "밀집 ms", "CSR ms"]
    write_tsv("c03-sparse", cols, rows, "bench/c03_sparse.py",
              f"float32 {N}x{N}, 행렬 벡터 곱, CSR 은 값 4바이트에 열 번호 4바이트",
              REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
