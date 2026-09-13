"""실험 7.1. 블룸 필터의 키당 비트와 오답률. 정리 7.2 의 예측과 실측.

키 백만 개를 넣고 없는 키 백만 개를 물어 있다고 답한 비율을 잰다.
해시 사전의 메모리와 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.bloom import BloomFilter, false_positive_rate, optimal_k  # noqa: E402

N = 1_000_000
BITS_PER_KEY = [4, 8, 12, 16, 24]


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = rng.choice(1 << 40, 2 * N, replace=False).astype(np.int64)
    inside, outside = keys[:N], keys[N:]
    set_mb = sys.getsizeof(set(inside.tolist())) / 1e6 + N * 28 / 1e6
    rows = [["해시 사전 (set)", 0, 0, set_mb, 0.0, 0.0]]
    for bpk in BITS_PER_KEY:
        m = bpk * N
        k = optimal_k(m, N)
        bf = BloomFilter(m, k)
        bf.add(inside)
        assert bf.contains(inside).all()
        fpr = bf.contains(outside).mean()
        rows.append([f"블룸 {bpk} 비트", bpk, k, bf.nbytes() / 1e6,
                     false_positive_rate(m, N, k) * 100, fpr * 100])
    cols = ["구조", "키당 비트", "해시 수", "MB", "오답 예측 퍼센트",
            "오답 실측 퍼센트"]
    cond = (f"키 {N:,}개를 넣고 없는 키 {N:,}개를 물음. "
            "해시 수는 (m/n) ln 2 로 반올림")
    write_tsv("c07-bloom", cols, rows, "bench/c07_bloom.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
