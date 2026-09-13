"""실험 7.3. MinHash 추정 오차와 LSH 띠의 S 곡선.

자카드를 알고 있는 문서 쌍을 만들어 서명 길이에 따른 추정 오차를 재고,
띠 나누기가 유사도에 따라 후보를 얼마나 잡는지 명제 7.14 와 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.minhash import (  # noqa: E402
    LSHIndex, candidate_probability, estimate_jaccard, jaccard, minhash)

PAIRS = 300
SET_SIZE = 500


def make_pair(rng: np.random.Generator, target: float) -> tuple[set[int], set[int]]:
    """자카드가 target 근처인 집합 쌍."""
    shared = int(SET_SIZE * 2 * target / (1 + target))
    base = rng.choice(1 << 40, SET_SIZE * 2, replace=False).tolist()
    common = set(base[:shared])
    a = common | set(base[shared:SET_SIZE])
    b = common | set(base[SET_SIZE:2 * SET_SIZE - shared])
    return a, b


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for k in [16, 64, 256]:
        errs = []
        for _ in range(PAIRS):
            a, b = make_pair(rng, rng.uniform(0.2, 0.9))
            j = jaccard(a, b)
            est = estimate_jaccard(minhash(a, k), minhash(b, k))
            errs.append(abs(est - j))
        rows.append([k, k * 8, float(np.mean(errs)), 0.5 / np.sqrt(k)])
    cols = ["서명 길이 k", "서명 바이트", "평균 절대 오차", "표준편차 상한"]
    write_tsv("c07-minhash", cols, rows, "bench/c07_minhash.py",
              f"집합 크기 {SET_SIZE}, 자카드 0.2 에서 0.9 인 쌍 {PAIRS}개", 1)

    rows2 = []
    bands, rows_per = 16, 4
    for target in [0.3, 0.5, 0.7, 0.8, 0.9]:
        hits = 0
        trials = 200
        for i in range(trials):
            a, b = make_pair(rng, target)
            idx = LSHIndex(bands, rows_per)
            idx.insert(0, minhash(a, bands * rows_per))
            hits += int(0 in idx.candidates(minhash(b, bands * rows_per)))
        pred = candidate_probability(target, bands, rows_per) * 100
        rows2.append([target, hits / trials * 100, pred])
    cols2 = ["자카드", "후보가 된 비율 퍼센트", "명제 예측 퍼센트"]
    write_tsv("c07-lsh", cols2, rows2, "bench/c07_minhash.py",
              f"띠 {bands}개, 띠마다 {rows_per}줄, 쌍 200개씩", 1)
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
