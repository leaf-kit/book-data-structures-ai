"""실험 1.1. 같은 숫자 백만 개, 세 가지 담는 법.

파이썬 리스트는 원소가 값이 아니라 객체를 가리키는 포인터다.
NumPy 배열은 원소가 값 그 자체다. 그 차이가 크기와 시간에 어떻게 나타나는지 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402

REPEAT = 7
N = 1_000_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    arr = rng.standard_normal(N).astype(np.float32)
    lst = arr.tolist()
    boxed = np.array(lst, dtype=object)

    obj_bytes = sum(sys.getsizeof(x) for x in lst[:1000]) * N // 1000
    size_list = sys.getsizeof(lst) + obj_bytes
    rows = [
        ["파이썬 리스트", size_list / 1e6, median_ms(lambda: sum(lst), REPEAT)],
        ["object 배열", (boxed.nbytes + 24 * N) / 1e6,
         median_ms(lambda: boxed.sum(), REPEAT)],
        ["float32 배열", arr.nbytes / 1e6, median_ms(lambda: arr.sum(), REPEAT)],
    ]
    cond = f"원소 {N:,}개, 리스트 크기는 포인터와 객체를 합한 추정"
    write_tsv("c01-boxed", ["담는 법", "크기 MB", "합산 ms"], rows,
              "bench/c01_boxed.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
