"""실험 3.2. 같은 밀도 50 퍼센트에서 CSR, 블록 희소, 2:4 의 바이트와 곱셈 시간.

0 의 자리를 약속하면 인덱스가 줄고 접근이 붙는다. 그 값을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.sparse import csr_matvec, dense_to_csr  # noqa: E402
from dsai.structured import (bsr_matvec, bytes_2of4, dense_to_bsr,  # noqa: E402
                             matvec_2of4, pack_2of4, prune_2of4)

REPEAT = 7
N = 4096
BLOCK = 32


def main() -> None:
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((N, N), dtype=np.float32)
    x = rng.standard_normal(N, dtype=np.float32)
    rows = []

    # 밀집
    t = median_ms(lambda: w @ x, REPEAT)
    rows.append(["밀집", w.nbytes / 1e6, t])

    # 2:4 로 가지치기한 것을 세 가지로 담는다
    p = prune_2of4(w)
    m = dense_to_csr(p)
    t = median_ms(lambda: csr_matvec(m, x), REPEAT)
    rows.append(["CSR (2:4 패턴)", m.nbytes() / 1e6, t])

    vals, meta = pack_2of4(p)
    t = median_ms(lambda: matvec_2of4(vals, meta, x), REPEAT)
    rows.append(["2:4 압축", bytes_2of4(N, N, 2) / 1e6, t])

    # 블록 희소. 블록 절반을 0 으로
    keep = rng.random((N // BLOCK, N // BLOCK)) < 0.5
    tile = np.ones((BLOCK, BLOCK), dtype=np.float32)
    wb = w * np.kron(keep, tile)
    b = dense_to_bsr(wb, BLOCK)
    t = median_ms(lambda: bsr_matvec(b, x), REPEAT)
    rows.append([f"블록 희소 {BLOCK}", b.nbytes() / 1e6, t])
    mb = dense_to_csr(wb)
    t = median_ms(lambda: csr_matvec(mb, x), REPEAT)
    rows.append(["CSR (블록 패턴)", mb.nbytes() / 1e6, t])

    write_tsv("c03-structured", ["표현", "MB", "곱셈 ms"], rows,
              "bench/c03_structured.py",
              f"float32 {N}x{N}, 0 이 아닌 원소 절반. 2:4 값은 float16", REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
