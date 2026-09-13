"""실험 3.3. 비트 수와 블록 크기에 따른 바이트와 오차.

정규분포 가중치에 이상값을 섞고, 텐서 전체 스케일과 블록 스케일의 오차를 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.quant import quant_error, quantize  # noqa: E402

N = 4096
CONFIGS = [(8, N * N), (8, 64), (4, N * N), (4, 64), (4, 32)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((N, N), dtype=np.float32) * 0.02
    # 이상값. 만 개 중 하나를 스무 배로
    outlier = rng.random((N, N)) < 1e-4
    w = np.where(outlier, w * 20, w)
    err16 = ((w.astype(np.float16).astype(np.float32) - w) ** 2).sum() / (w * w).sum()
    rows = [["float32", 32, 0, w.nbytes / 1e6, 0.0],
            ["float16", 16, 0, w.nbytes / 2 / 1e6, float(err16) * 100]]
    for bits, block in CONFIGS:
        q = quantize(w, bits, block)
        name = f"int{bits} " + ("텐서 하나" if block == N * N else f"블록 {block}")
        rows.append([name, bits, 0 if block == N * N else block, q.nbytes() / 1e6,
                     quant_error(w, q) * 100])
    cols = ["표현", "비트", "블록", "MB", "상대 오차 퍼센트"]
    write_tsv("c03-quant", cols, rows, "bench/c03_quant.py",
              f"{N}x{N} 정규분포 가중치, 만 개 중 하나가 스무 배 이상값. "
              "오차는 상대 제곱 오차의 퍼센트", 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
