"""실험 4.1. 포인터 추적. 순차 고리와 무작위 고리, 그리고 독립된 고리 여럿.

한 걸음의 값은 다음 주소를 아는 시점이 정한다.
무작위 고리는 걸음마다 라인 하나를 기다린다.
독립된 고리 64 개를 같이 가면 기다림이 겹쳐서 걸음당 값이 준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.linked import chain_random, chain_sequential, chase, chase_many  # noqa: E402

REPEAT = 5
N = 8 * 1024 * 1024          # 노드 800만 개, 64MB 의 next 배열
STEPS = 1_000_000
K = 4096
STEPS_MANY = 4_000_000


def main() -> None:
    seq = chain_sequential(N)
    rnd = chain_random(N, SEED)
    starts = np.random.default_rng(SEED + 1).integers(0, N, K)
    rows = []
    t = median_ms(lambda: chase(seq, 0, STEPS), REPEAT)
    rows.append(["순차 고리, 하나", STEPS, t, t * 1e6 / STEPS])
    t = median_ms(lambda: chase(rnd, 0, STEPS), REPEAT)
    rows.append(["무작위 고리, 하나", STEPS, t, t * 1e6 / STEPS])
    t = median_ms(lambda: chase_many(seq, starts, STEPS_MANY // K), REPEAT)
    rows.append([f"순차 고리, {K}개 같이", STEPS_MANY, t, t * 1e6 / STEPS_MANY])
    t = median_ms(lambda: chase_many(rnd, starts, STEPS_MANY // K), REPEAT)
    rows.append([f"무작위 고리, {K}개 같이", STEPS_MANY, t, t * 1e6 / STEPS_MANY])
    cols = ["고리", "걸음 수", "시간 ms", "걸음당 ns"]
    cond = (f"노드 {N:,}개 (next 배열 64MB). 하나는 {STEPS:,}걸음, "
            f"같이 가는 고리는 합쳐 {STEPS_MANY:,}걸음")
    write_tsv("c04-chase", cols, rows, "bench/c04_chase.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
