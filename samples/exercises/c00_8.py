"""문제 0.8. 질의 여러 개를 한 번에 받는 전수 탐색."""
import numpy as np

from dsai.knn import knn_bruteforce, random_corpus
from dsai.roofline import intensity


def knn_batch(xs: np.ndarray, qs: np.ndarray, k: int) -> np.ndarray:
    """qs 는 (m, d). 행렬 곱 하나로 m 개 질의의 점수를 구한다."""
    scores = xs @ qs.T
    top = np.argpartition(-scores, k, axis=0)[:k]
    out = np.empty_like(top)
    for j in range(qs.shape[0]):
        col = top[:, j]
        out[:, j] = col[np.argsort(-scores[col, j])]
    return out.T


def batch_intensity(n: int, d: int, m: int, elem_bytes: int = 4) -> float:
    flops = 2 * n * d * m
    nbytes = (n * d + d * m + n * m) * elem_bytes
    return intensity(flops, nbytes)


def test_batch_matches_single():
    xs = random_corpus(3000, 32, 2)
    qs = random_corpus(8, 32, 3)
    got = knn_batch(xs, qs, 5)
    for j in range(8):
        assert np.array_equal(got[j], knn_bruteforce(xs, qs[j], 5))


def test_batch_intensity_grows():
    assert batch_intensity(10**6, 128, 64) > 30 * batch_intensity(10**6, 128, 1)
