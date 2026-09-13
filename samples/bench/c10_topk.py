"""실험 10.1. 상위 k 개 고르기. 전부 정렬, 크기 k 힙, 부분 정렬 (선택).

어휘 크기 V = 128,000 의 로짓에서 k = 50 을 고른다.
파이썬 힙은 느리므로 V 를 줄여 비율만 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.heap import topk_heap, topk_select, topk_sort  # noqa: E402

V = 128_000
K = 50


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(V)
    a = set(topk_sort(x, K).tolist())
    assert a == set(topk_select(x, K).tolist()) == set(topk_heap(x, K).tolist())
    rows = []
    rows.append(["전부 정렬 (NumPy)", V, K, median_ms(lambda: topk_sort(x, K), 5)])
    rows.append(["부분 정렬 (NumPy argpartition)", V, K,
                 median_ms(lambda: topk_select(x, K), 5)])
    rows.append(["크기 k 힙 (파이썬)", V, K, median_ms(lambda: topk_heap(x, K), 3)])
    small = x[:8000]
    rows.append(["전부 정렬 (파이썬 sorted)", 8000, K,
                 median_ms(lambda: sorted(small.tolist(), reverse=True)[:K], 5)])
    rows.append(["크기 k 힙 (파이썬), V=8000", 8000, K,
                 median_ms(lambda: topk_heap(small, K), 5)])
    cols = ["방식", "V", "k", "ms"]
    cond = ("표준 정규 로짓. 세 방법의 상위 50 개 집합이 같음을 확인. "
            "다섯 번 중앙값 (파이썬 힙은 셋)")
    write_tsv("c10-topk", cols, rows, "bench/c10_topk.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
