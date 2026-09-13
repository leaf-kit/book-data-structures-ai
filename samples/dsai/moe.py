"""3장. 전문가 혼합의 gather 와 scatter. 토큰마다 다른 행렬을 읽는 작업량.

밀집 층은 모든 토큰이 같은 행렬을 읽는다.
전문가 혼합은 토큰마다 라우터가 고른 전문가 행렬 하나를 읽는다.
토큰을 전문가별로 모으고 (gather), 계산하고, 제자리로 흩는다 (scatter).
"""
from __future__ import annotations

import numpy as np


def route_top1(logits: np.ndarray) -> np.ndarray:
    """정의 3.6 의 라우터. 토큰마다 점수가 가장 큰 전문가 하나."""
    return np.argmax(logits, axis=1)


def moe_forward(x: np.ndarray, experts: np.ndarray, assign: np.ndarray) -> np.ndarray:
    """알고리즘 3.4. 전문가별로 토큰을 모아 곱하고 제자리로 돌린다.

    x 는 (B, d), experts 는 (E, d, h), assign 은 (B,). 결과는 (B, h).
    """
    out = np.empty((x.shape[0], experts.shape[2]), dtype=x.dtype)
    for e in np.unique(assign):
        idx = np.nonzero(assign == e)[0]          # gather 할 자리
        out[idx] = x[idx] @ experts[e]           # 전문가 e 의 행렬을 한 번 읽는다
    return out


def dense_forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w


def experts_touched_expected(num_experts: int, batch: int) -> float:
    """명제 3.4. 배치 B 가 균등 라우팅될 때 한 번이라도 뽑히는 전문가 수의 기대값."""
    return num_experts * (1.0 - (1.0 - 1.0 / num_experts) ** batch)


def moe_intensity(batch: int, d: int, h: int, num_experts: int,
                  elem_bytes: int) -> float:
    """토큰 B 개 처리의 연산 밀도. 읽는 전문가 수가 기대값이라고 본다."""
    flops = 2 * batch * d * h
    touched = experts_touched_expected(num_experts, batch)
    nbytes = touched * d * h * elem_bytes + batch * (d + h) * elem_bytes
    return flops / nbytes
