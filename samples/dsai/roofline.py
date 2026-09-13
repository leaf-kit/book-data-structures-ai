"""0장. 루프라인 모델. 연산 밀도로 병목을 읽는다.

연산 하나가 바이트 하나를 옮겨 와서 몇 번 계산하는가. 그 비율이 연산 밀도 I 다.
같은 기계에서 밀도가 낮은 연산은 대역폭에, 높은 연산은 연산 장치에 막힌다.
"""
from __future__ import annotations

import numpy as np


def dot(x: np.ndarray, y: np.ndarray) -> float:
    """내적. 원소마다 곱 하나 합 하나, 바이트는 두 벡터를 한 번 읽는다."""
    return float(x @ y)


def matvec(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """행렬 벡터 곱. 행렬을 한 번 읽고 원소마다 두 번 계산한다."""
    return a @ x


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행렬 행렬 곱. 읽는 바이트에 비해 계산이 n 배 많다."""
    return a @ b


def flops_dot(n: int) -> int:
    return 2 * n


def flops_matvec(m: int, n: int) -> int:
    return 2 * m * n


def flops_matmul(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def bytes_dot(n: int, elem_bytes: int) -> int:
    return 2 * n * elem_bytes


def bytes_matvec(m: int, n: int, elem_bytes: int) -> int:
    return (m * n + n + m) * elem_bytes


def bytes_matmul(m: int, k: int, n: int, elem_bytes: int) -> int:
    return (m * k + k * n + m * n) * elem_bytes


def intensity(flops: int, nbytes: int) -> float:
    """연산 밀도 I = W / M. 단위는 FLOP/byte."""
    return flops / nbytes


def attainable(peak_gflops: float, bandwidth_gbs: float, i: float) -> float:
    """정의 0.3. 달성 가능한 처리량은 두 지붕 중 낮은 쪽이다."""
    return min(peak_gflops, bandwidth_gbs * i)


def ridge_point(peak_gflops: float, bandwidth_gbs: float) -> float:
    """두 지붕이 만나는 밀도. 이보다 낮으면 대역폭에, 높으면 연산에 막힌다."""
    return peak_gflops / bandwidth_gbs
