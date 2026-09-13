"""문제 14.5. 정수 하나로 묶지 않고 lexsort 로 두 순위를 정렬하는 배가 정렬."""
import numpy as np

from dsai.suffix import build_suffix_array


def build_suffix_array_lexsort(text: np.ndarray) -> np.ndarray:
    n = len(text)
    _, rank = np.unique(text, return_inverse=True)
    rank = rank.astype(np.int64)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        second[: n - k] = rank[k:]
        sa = np.lexsort((second, rank))                 # 뒤 열이 첫째 키
        changed = np.ones(n, dtype=bool)
        changed[1:] = ((rank[sa[1:]] != rank[sa[:-1]])
                       | (second[sa[1:]] != second[sa[:-1]]))
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.cumsum(changed) - 1
        rank = new_rank
        if rank.max() == n - 1:
            return sa
        k *= 2


def test_lexsort_version_agrees():
    rng = np.random.default_rng(0)
    for size in (7, 100, 5000):
        t = rng.integers(0, 5, size).astype(np.int64)
        assert np.array_equal(build_suffix_array_lexsort(t), build_suffix_array(t))
