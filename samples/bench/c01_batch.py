"""실험 1.5. 배치 크기에 따른 밀도와 처리량.

같은 행렬에 질의 B 개를 한 번에 곱한다.
B 가 커질수록 밀도가 오르고 처리량이 지붕에 다가간다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.batch import intensity_batched, scores_batch  # noqa: E402

REPEAT = 7
N, D = 262_144, 128  # 128MB
BATCHES = [1, 4, 16, 64, 256]


def main() -> None:
    rng = np.random.default_rng(SEED)
    xs = rng.standard_normal((N, D), dtype=np.float32)
    rows = []
    for b in BATCHES:
        qs = rng.standard_normal((b, D), dtype=np.float32)
        t = median_ms(lambda: scores_batch(xs, qs), REPEAT)
        gflops = 2 * N * D * b / (t / 1000) / 1e9
        rows.append([b, intensity_batched(N, D, b), gflops, t / b])
    cond = f"float32, 행렬 {N:,}x{D} (128MB), 밀도는 FLOP/byte"
    write_tsv("c01-batch", ["배치", "밀도", "처리량", "질의당 ms"], rows,
              "bench/c01_batch.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
