"""실험 15.3. 고정 크기 상태. 재귀와 스캔의 시간, 그리고 어텐션 상태와의 메모리 비교.

길이를 올리며 한 단계씩 도는 재귀와 log n 단계의 스캔의 시간을 재고,
같은 길이의 KV 캐시 크기를 나란히 적는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.ssm import (attention_state_bytes, run_recurrent, run_scan,  # noqa: E402
                      ssm_state_bytes)

S = 64                    # 상태 크기
LAYERS, HEADS, D = 32, 32, 128


def main() -> None:
    rng = np.random.default_rng(SEED)
    a = np.exp(-rng.uniform(0.01, 0.5, S))
    b = rng.standard_normal(S)
    c = rng.standard_normal(S) / S
    rows = []
    for n in [4096, 16384, 65536]:
        x = rng.standard_normal(n)
        y1 = run_recurrent(x, a, b, c)
        y2 = run_scan(x, a, b, c)
        assert np.allclose(y1, y2, atol=1e-6)
        ms_r = median_ms(lambda: run_recurrent(x, a, b, c), 3)
        ms_s = median_ms(lambda: run_scan(x, a, b, c), 3)
        rows.append([n, ms_r, ms_s, int(np.ceil(np.log2(n))),
                     ssm_state_bytes(LAYERS, D * HEADS, S, 2) / 1e6,
                     attention_state_bytes(n, LAYERS, HEADS, D, 2) / 1e6])
    cols = ["길이 n", "재귀 ms", "스캔 ms", "스캔 단계", "고정 상태 MB", "KV 캐시 MB"]
    cond = (f"상태 크기 {S}, 대각 A. "
            f"메모리는 층 {LAYERS}, 헤드 {HEADS}, 차원 {D}, 2 바이트의 모델 크기로 셈. "
            "두 방법의 출력이 같음을 확인. 재귀는 단계 n 개, 스캔은 단계 log n 개. "
            "세 번 중앙값")
    write_tsv("c15-ssm", cols, rows, "bench/c15_ssm.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
