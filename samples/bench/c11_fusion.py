"""실험 11.3. 원소별 사슬 다섯 개. 하나씩과 블록 융합의 시간과 이동량.

배열 3,200 만 개 (256 MB) 에서 연산 다섯 개를 하나씩 하는 것과
블록마다 사슬 전체를 도는 것.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.fusion import (  # noqa: E402
    CHAIN_CHEAP, CHAIN_HEAVY, fused_blocks, traffic_bytes, unfused)

N = 32_000_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = np.abs(rng.standard_normal(N)) + 0.1
    out = np.empty_like(x)
    rows = []
    chains = [("싼 사슬", CHAIN_CHEAP), ("비싼 사슬", CHAIN_HEAVY)]
    for name, chain in chains:
        ref = unfused(x, chain)
        ms0 = median_ms(lambda: unfused(x, chain), 3)
        gb0 = traffic_bytes(N, len(chain), 8, False) / 1e9
        rows.append([name, "하나씩", 0, ms0, gb0, 1.0])
        for block in [4_096, 65_536, 1_048_576]:
            assert np.allclose(fused_blocks(x, out, block, chain), ref)
            ms = median_ms(lambda: fused_blocks(x, out, block, chain), 3)
            rows.append([name, f"블록 {block // 1024}K", block, ms,
                         traffic_bytes(N, len(chain), 8, True) / 1e9, ms0 / ms])
    cols = ["사슬", "방식", "블록 원소", "ms", "명제의 이동 GB", "배속"]
    cond = (f"float64 {N:,}개, 원소별 연산 다섯 개의 사슬 둘. "
            "이동량은 명제의 값이고 실측이 아니다. 세 번 중앙값")
    write_tsv("c11-fusion", cols, rows, "bench/c11_fusion.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
