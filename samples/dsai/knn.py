"""0장. 이 책이 끝까지 다시 보는 문제, 최근접 탐색.

벡터 n 개 중에서 질의와 가장 가까운 k 개를 찾는다. 여기서는 전부 훑는 정확 탐색만 둔다.
5부에서 이 함수의 답을 정답으로 삼아 근사 인덱스의 재현율을 잰다.
"""
from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """각 행을 길이 1 로 만든다. 그러면 내적이 코사인 유사도가 된다."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def knn_bruteforce(xs: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """정확 탐색. n 개 전부와 내적을 구하고 큰 순서로 k 개를 고른다.

    xs 는 (n, d), q 는 (d,). 둘 다 normalize 를 거쳤다고 본다.
    돌려주는 것은 유사도가 큰 순서로 정렬된 인덱스 k 개다.
    """
    scores = xs @ q
    if k >= len(scores):
        return np.argsort(-scores)
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """정의 0.5. 찾은 k 개 중 정답 k 개에 든 것의 비율."""
    return len(set(found.tolist()) & set(truth.tolist())) / len(truth)


def random_corpus(n: int, d: int, seed: int) -> np.ndarray:
    """임베딩 자리에 놓는 대역. 진짜 임베딩은 6장과 9장에서 만든다."""
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal((n, d), dtype=np.float32))
