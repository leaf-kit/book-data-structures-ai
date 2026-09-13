"""실험 0.3. 정확 최근접 탐색의 지연은 말뭉치 크기에 비례한다.

이 표가 5부 전체의 출발점이다.
여기서 잰 지연을 근사 인덱스가 얼마나 줄이는지를 재게 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.knn import knn_bruteforce, random_corpus  # noqa: E402

REPEAT = 9
D = 128
K = 10
SIZES = [1_000, 10_000, 100_000, 1_000_000]


def main() -> None:
    rows = []
    q = random_corpus(1, D, SEED + 1)[0]
    for n in SIZES:
        xs = random_corpus(n, D, SEED)
        t = median_ms(lambda: knn_bruteforce(xs, q, K), REPEAT)
        rows.append([n, round(n * D * 4 / 1e6, 1), t, n / (t / 1000) / 1e6])
    write_tsv("c00-bruteforce", ["문서 수", "크기 MB", "지연 ms", "초당 M벡터"], rows,
              "bench/c00_knn.py", f"float32, 차원 {D}, k={K}, 질의 하나", REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
