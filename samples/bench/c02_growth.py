"""실험 2.1. 성장 방법에 따라 옮긴 바이트와 시간이 어떻게 다른가.

원소 백만 개를 하나씩 넣는다.
두 배 늘리기, 1.125 배 늘리기, 고정 증가, 그리고 미리 잡기.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import median_ms, write_tsv  # noqa: E402
from dsai.dynarray import DynArray  # noqa: E402

REPEAT = 5
N = 1_000_000


def fill(factor: float = 2.0, increment: int = 0) -> DynArray:
    a = DynArray(factor=factor, increment=increment)
    for i in range(N):
        a.append(i)
    return a


def fill_prealloc() -> np.ndarray:
    a = np.empty(N, dtype=np.float32)
    for i in range(N):
        a[i] = i
    return a


def main() -> None:
    rows = []
    for name, kw in [("두 배 늘리기", dict(factor=2.0)),
                     ("1.5배 늘리기", dict(factor=1.5)),
                     ("1.125배 늘리기", dict(factor=1.125)),
                     ("1024씩 더하기", dict(increment=1024))]:
        a = fill(**kw)
        t = median_ms(lambda: fill(**kw), REPEAT)
        rows.append([name, a.grow_count, a.copied_bytes / 1e6,
                     a.waste_bytes() / 1e6, t])
    t = median_ms(fill_prealloc, REPEAT)
    rows.append(["미리 잡기", 0, 0.0, 0.0, t])
    cols = ["방법", "옮긴 횟수", "옮긴 MB", "남은 MB", "시간 ms"]
    write_tsv("c02-growth", cols, rows, "bench/c02_growth.py",
              f"float32 원소 {N:,}개를 하나씩 추가", REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
