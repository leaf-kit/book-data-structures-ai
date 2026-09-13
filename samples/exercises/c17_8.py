"""문제 17.8. 성적표의 판정 하나를 코드로 다시 확인한다.
정렬 배열의 이진 탐색은 그대로 유효한가.
"""
import numpy as np

from dsai.layout import binary_search, eytzinger, eytzinger_search


def test_binary_search_still_valid_and_eytzinger_matches():
    rng = np.random.default_rng(1)
    keys = np.sort(rng.choice(10_000_000, 100_000, replace=False))
    tree = eytzinger(keys)
    for q in keys[rng.integers(0, len(keys), 50)]:
        i, _ = binary_search(keys, int(q))
        assert keys[i] == q
        j, _ = eytzinger_search(tree, int(q))
        assert tree[j] == q
