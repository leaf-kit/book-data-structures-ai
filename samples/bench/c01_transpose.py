"""실험 1.2 와 1.4. 뷰는 공짜이고, 뷰를 옮기는 값은 순서가 정한다.

전치 뷰를 만드는 시간, 연속 복사, 전치 복사, 블록 전치를 같은 행렬로 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.tiling import transpose_blocked, transpose_copy  # noqa: E402

REPEAT = 5
N = 8192  # 8192 x 8192 float32 = 256MB
BLOCKS = [16, 64, 256, 1024]


def main() -> None:
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((N, N), dtype=np.float32)
    mb = a.nbytes / 1e6

    # 출력 버퍼를 미리 잡아 둔다. 새로 잡으면 페이지 폴트가 섞인다
    out = np.empty_like(a)

    t_view = median_ms(lambda: a.T, 101)
    t_copy = median_ms(lambda: np.copyto(out, a), REPEAT)
    t_tcopy = median_ms(lambda: transpose_copy(a, out), REPEAT)
    rows = [
        ["전치 뷰 만들기", 0.0, t_view, 0.0],
        ["연속 복사", mb, t_copy, 2 * mb / t_copy],
        ["전치 복사", mb, t_tcopy, 2 * mb / t_tcopy],
    ]
    for b in BLOCKS:
        t = median_ms(lambda: transpose_blocked(a, b, out), REPEAT)
        rows.append([f"블록 전치 {b}", mb, t, 2 * mb / t])
    cond = f"float32 {N}x{N} (256MB). 대역폭은 읽기와 쓰기 합 GB/s"
    write_tsv("c01-transpose", ["연산", "옮긴 MB", "시간 ms", "대역폭"], rows,
              "bench/c01_transpose.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
