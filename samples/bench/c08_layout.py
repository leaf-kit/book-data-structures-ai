"""실험 8.4. 같은 정렬된 배열, 세 탐색. 이진 탐색, 아이칭거 배치, 배치 정렬 탐색.

키 천만 개에서 질의 십만 개를 찾는 시간을 잰다.
하나씩 찾는 둘은 2,000개로 재서 질의당 시간으로 적는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.layout import (  # noqa: E402
    batch_search, binary_search, eytzinger, eytzinger_search)

N = 10_000_000
Q = 100_000
Q_ONE = 2_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = np.sort(rng.choice(1 << 40, N, replace=False).astype(np.int64))
    tree = eytzinger(keys)
    queries = rng.choice(keys, Q)
    q1 = queries[:Q_ONE]
    kl = keys.tolist()
    tl = tree.tolist()

    def bin_one():
        return [binary_search(kl, int(q)) for q in q1]

    def eyt_one():
        return [eytzinger_search(tl, int(q)) for q in q1]

    def np_one():
        return [int(np.searchsorted(keys, q)) for q in q1]

    def batch():
        return batch_search(keys, queries)

    def unsorted_batch():
        return np.searchsorted(keys, queries)

    rows = []
    for name, fn, cnt in [("이진 탐색 (파이썬)", bin_one, Q_ONE),
                          ("아이칭거 (파이썬)", eyt_one, Q_ONE),
                          ("searchsorted 하나씩", np_one, Q_ONE),
                          ("searchsorted 배치 (정렬 안 함)", unsorted_batch, Q),
                          ("배치 정렬 탐색", batch, Q)]:
        ms = median_ms(fn, 5)
        rows.append([name, cnt, ms, ms / cnt * 1e3])
    # 정답 확인
    assert np.array_equal(batch(), unsorted_batch())
    assert all(keys[binary_search(kl, int(q))[0]] == q for q in q1[:100])
    assert all(tree[eytzinger_search(tl, int(q))[0]] == q for q in q1[:100])
    cols = ["방식", "질의 수", "ms", "질의당 us"]
    cond = (f"정렬된 40비트 키 {N:,}개. 파이썬 루프 둘은 리스트로 바꿔 돌렸다. "
            "다섯 번 중앙값")
    write_tsv("c08-layout", cols, rows, "bench/c08_layout.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
