"""실험 15.1. 어텐션의 점수 행렬을 메모리에 내는 것과 타일로 없애는 것.
길이에 따른 시간과 메모리.

질의와 키 n 개, 차원 64. n 을 1,024 에서 16,384 까지 올리며
두 방법의 시간과 점수 행렬의 크기를 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.attention import (attention_naive, attention_tiled,  # noqa: E402
                            traffic_naive, traffic_tiled)

D = 64


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [1024, 4096, 16384]:
        q = rng.standard_normal((n, D)).astype(np.float32)
        k = rng.standard_normal((n, D)).astype(np.float32)
        v = rng.standard_normal((n, D)).astype(np.float32)
        ref = attention_naive(q, k, v)
        assert np.allclose(attention_tiled(q, k, v, 1024), ref, atol=1e-3)
        ms_n = median_ms(lambda: attention_naive(q, k, v), 3)
        ms_t = median_ms(lambda: attention_tiled(q, k, v, 1024), 3)
        rows.append([n, n * n * 4 / 1e6, traffic_naive(n, n, D, D, 4) / 1e6,
                     traffic_tiled(n, n, D, D, 4, 1024) / 1e6, ms_n, ms_t])
    cols = ["길이 n", "점수 행렬 MB", "하나씩 이동 MB", "타일 이동 MB", "하나씩 ms",
            "타일 ms"]
    cond = f"차원 {D}, float32, 타일 1,024. 두 방법의 출력이 같음을 확인. 세 번 중앙값"
    write_tsv("c15-attention", cols, rows, "bench/c15_attention.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
