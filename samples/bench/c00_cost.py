"""실험 0.1. 같은 n 번 접근, 순차와 무작위.

균일 비용 모델은 둘의 값이 같다고 말한다. 재 보면 다르다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.cost import gather_sum, random_index, sequential_index  # noqa: E402

REPEAT = 7
SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]


def main() -> None:
    rows = []
    for n in SIZES:
        values = np.arange(n, dtype=np.float32)
        seq = sequential_index(n)
        rnd = random_index(n, SEED)
        t_seq = median_ms(lambda: gather_sum(values, seq), REPEAT)
        t_rnd = median_ms(lambda: gather_sum(values, rnd), REPEAT)
        mb = n * 4 / 1e6
        rows.append([n, round(mb, 1), t_seq, t_rnd, t_rnd / t_seq])
    write_tsv("c00-seq-vs-random",
              ["원소 수", "크기 MB", "순차 ms", "무작위 ms", "비율"], rows,
              "bench/c00_cost.py", "float32, 원소마다 한 번씩 읽어 더함", REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
