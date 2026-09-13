"""실험 14.1. 접미사 배열 만들기와 패턴 세기. 9장의 말뭉치를 토큰 열로.

토큰 백만 개의 접미사 배열을 만들고,
패턴 세기를 접미사 배열과 순차 훑기와 n-그램 dict 로 견준다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version  # noqa: E402
from bench._util import median_ms, write_tsv  # noqa: E402
from dsai.suffix import build_suffix_array, count  # noqa: E402

N = 1_000_000


def main() -> None:
    # 바이트가 토큰
    text = np.frombuffer(stdlib_text(N), dtype=np.uint8).astype(np.int64)
    t0 = time.perf_counter()
    sa = build_suffix_array(text)
    build_s = time.perf_counter() - t0
    raw = bytes(text.astype(np.uint8))
    patterns = [b"def ", b"return", b"self.", b"import numpy", b"raise ValueError("]
    rows = []
    for p in patterns:
        pat = np.frombuffer(p, dtype=np.uint8).astype(np.int64)
        c_sa = count(text, sa, pat)
        c_scan = raw.count(p) if not p.endswith(b"(") else raw.count(p)
        assert c_sa >= c_scan - 5 and c_sa <= c_scan + len(p)   # 겹치는 것의 차이만
        ms_sa = median_ms(lambda: count(text, sa, pat), 5)
        ms_scan = median_ms(lambda: raw.count(p), 5)
        rows.append([p.decode(), len(p), c_sa, ms_sa * 1e3, ms_scan * 1e3])
    cols = ["패턴", "길이", "개수", "접미사 배열 us", "순차 훑기 us"]
    cond = (f"{version()} 표준 라이브러리 소스 {N:,} 바이트, 바이트가 토큰. "
            f"접미사 배열 만들기 {build_s:.1f}초 (배가 정렬), "
            f"배열 {sa.nbytes / 1e6:.0f} MB. "
            "순차 훑기는 C 로 짜인 bytes.count")
    write_tsv("c14-suffix", cols, rows, "bench/c14_suffix.py", cond, 5)
    for r in rows:
        print(r)
    print(cond)


if __name__ == "__main__":
    main()
