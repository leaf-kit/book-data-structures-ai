"""실험 6.4. 키 천만 개를 세는 두 길. 하나씩 넣기와 정렬해서 세기.
그리고 한꺼번에 넣을 때의 충돌.

같은 키 열을 dict 에 하나씩 넣는 것과, 정렬해 경계를 세는 것을 잰다.
그리고 키 전부를 한 걸음에 자기 자리에 쓰려 할 때 몇 걸음이 드는지 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.hashing import random_odd  # noqa: E402
from dsai.parhash import batch_insert_rounds, count_serial, count_sorted  # noqa: E402

REPEAT = 3
SIZES = [100_000, 1_000_000, 10_000_000]
UNIQUE = 100_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in SIZES:
        keys = rng.integers(0, UNIQUE, n).astype(np.int64)
        t_serial = median_ms(lambda: count_serial(keys), REPEAT)
        t_sorted = median_ms(lambda: count_sorted(keys), REPEAT)
        rows.append([n, t_serial, t_sorted, t_serial / t_sorted])
    cols = ["키 수", "하나씩 dict ms", "정렬해 세기 ms", "비율"]
    cond = f"서로 다른 키 {UNIQUE:,}개에서 뽑은 정수 키. 세 번 중앙값"
    write_tsv("c06-parallel", cols, rows, "bench/c06_parallel.py", cond, REPEAT)

    rows2 = []
    a = random_odd(rng)
    for alpha in [0.25, 0.5, 0.75]:
        m = 1 << 20
        ks = rng.choice(1 << 32, int(m * alpha), replace=False).astype(np.int64)
        rows2.append([alpha, batch_insert_rounds(ks, m, a)])
    cond2 = "자리 1,048,576 개에 키 전부를 한꺼번에 넣을 때 다 들어가기까지의 걸음"
    write_tsv("c06-rounds", ["적재율", "걸음 수"], rows2, "bench/c06_parallel.py",
              cond2, 1)
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
