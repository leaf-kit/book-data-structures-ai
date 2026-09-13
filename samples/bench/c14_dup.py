"""실험 14.3. 말뭉치 안의 긴 중복. 길이 문턱에 따라 중복이 덮는 비율.

표준 라이브러리 소스 백만 바이트에서 길이 50, 100, 200,
400 바이트 이상의 반복 구간이 덮는 비율을 잰다.
7장의 문서 단위 중복 제거와 견주기 위해 문서 (파일) 단위의 정확 중복도 센다.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version  # noqa: E402
from bench._util import write_tsv  # noqa: E402
from dsai.dupspan import (covered_tokens, duplicated_positions,  # noqa: E402
                          longest_repeat)
from dsai.suffix import build_suffix_array, lcp_array  # noqa: E402

N = 1_000_000


def main() -> None:
    text = np.frombuffer(stdlib_text(N), dtype=np.uint8).astype(np.int64)
    sa = build_suffix_array(text)
    t0 = time.perf_counter()
    lcp = lcp_array(text, sa)
    lcp_s = time.perf_counter() - t0
    rows = []
    for L in [50, 100, 200, 400]:
        pos = duplicated_positions(text, sa, L)
        cov = covered_tokens(pos, L, len(text))
        rows.append([L, len(pos), cov / len(text) * 100])
    ln, a, b = longest_repeat(text, sa)
    cols = ["길이 문턱 바이트", "중복 시작 위치 수", "덮는 비율 퍼센트"]
    cond = (f"{version()} 표준 라이브러리 소스 {N:,} 바이트. LCP 배열 {lcp_s:.1f}초. "
            f"가장 긴 반복은 {ln:,} 바이트 (위치 {a:,} 와 {b:,}). "
            f"lcp 의 최대 {int(lcp.max()):,}")
    write_tsv("c14-dup", cols, rows, "bench/c14_dup.py", cond, 1)
    for r in rows:
        print(r)
    print(cond)


if __name__ == "__main__":
    main()
