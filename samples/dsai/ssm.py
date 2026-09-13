"""15장. 고정 크기 상태. 선형 재귀와 상태 공간.

어텐션의 상태는 KV 캐시이고 길이에 비례해 큰다. 선형 재귀는 상태가 고정 크기다.
h_t = A h_{t-1} + B x_t, y_t = C h_t. 한 단계가 상태 크기의 곱셈이고 길이와 무관하다.
같은 식을 log n 단계로 계산하는 것이 5장의 누적합과 같은 스캔이다. 학습은 스캔,
추론은 재귀다.
"""
from __future__ import annotations

import numpy as np


def recurrent_step(h: np.ndarray, x: np.ndarray, a: np.ndarray,
                   b: np.ndarray) -> np.ndarray:
    """정의 15.4. 대각 A 의 한 단계. h, a, b 는 상태 크기 s. 곱셈 2s 개."""
    return a * h + b * x


def run_recurrent(x: np.ndarray, a: np.ndarray, b: np.ndarray,
                  c: np.ndarray) -> np.ndarray:
    """디딤돌. 길이 n 의 입력을 한 단계씩. 상태는 언제나 s 개다."""
    n = len(x)
    h = np.zeros(len(a))
    y = np.empty(n)
    for t in range(n):
        h = recurrent_step(h, x[t], a, b)
        y[t] = c @ h
    return y


def run_scan(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """알고리즘 15.3. 결합 법칙으로 푸는 스캔. 5장의 누적합과 같은 두 배씩 건너뛰기.
    (A, B) 뒤에 (A', B') 를 붙이면 (A A', A' B + B') 이다.
    log2 n 단계이고 단계마다 배열 전체를 만진다.
    """
    n = len(x)
    A = np.tile(a, (n, 1))                    # (n, s). 원소 t 는 h_t = A h + B
    B = b[None, :] * x[:, None]
    step = 1
    while step < n:
        A_new, B_new = A.copy(), B.copy()
        A_new[step:] = A[:-step] * A[step:]
        B_new[step:] = A[step:] * B[:-step] + B[step:]
        A, B = A_new, B_new
        step *= 2
    return B @ c


def attention_state_bytes(n: int, layers: int, heads: int, d: int,
                          itemsize: int) -> int:
    """KV 캐시. 토큰마다 층 x 헤드 x 차원 x 2."""
    return n * layers * heads * d * 2 * itemsize


def ssm_state_bytes(layers: int, channels: int, s: int, itemsize: int) -> int:
    """선형 재귀의 상태. 길이와 무관."""
    return layers * channels * s * itemsize
