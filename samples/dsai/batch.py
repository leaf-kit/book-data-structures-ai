"""1장. 배치와 브로드캐스팅. 차원을 하나 더하면 밀도가 오른다.

질의 하나씩 처리하면 행렬을 질의마다 다시 읽는다.
질의 B 개를 묶으면 행렬을 한 번 읽고 B 번 쓴다.
"""
from __future__ import annotations

import numpy as np


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """정의 1.4 의 규칙. 뒤에서부터 맞추고, 1 은 늘어나고, 다르면 실패한다."""
    out = []
    for x, y in zip(reversed((1,) * (len(b) - len(a)) + a),
                    reversed((1,) * (len(a) - len(b)) + b)):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"맞지 않는 모양 {a} 와 {b}")
    return tuple(reversed(out))


def scores_one(xs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """질의 하나. 행렬 벡터 곱. 밀도 0.5."""
    return xs @ q


def scores_batch(xs: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """질의 B 개. 행렬 행렬 곱. xs 를 한 번 읽어 B 번 쓴다."""
    return xs @ qs.T


def intensity_batched(n: int, d: int, b: int, elem_bytes: int = 4) -> float:
    """명제 1.3. 배치 크기 b 에서의 연산 밀도."""
    flops = 2 * n * d * b
    nbytes = (n * d + d * b + n * b) * elem_bytes
    return flops / nbytes
