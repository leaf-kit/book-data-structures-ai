"""15장. 연관 배열로서의 어텐션, 타일링과 온라인 소프트맥스.

어텐션은 키 n 개에 값 n 개를 둔 표에서 질의로 값을 읽는 것이다.
해시 테이블은 키 하나가 값 하나를 주고,
어텐션은 모든 키가 유사도만큼 값을 준다. 소프트맥스 가중 합이다.
행렬 전체를 만들면 n x n 이 메모리에 나간다.
타일마다 최대와 합을 갱신하는 온라인 소프트맥스로 그것을 없앤다.
"""
from __future__ import annotations

import numpy as np


def attention_naive(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """정의 15.1. (m, d) 질의, (n, d) 키, (n, dv) 값. 점수 행렬 (m,
    n) 을 메모리에 만든다.
    """
    s = q @ k.T / np.sqrt(q.shape[1])
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def attention_tiled(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    block: int) -> np.ndarray:
    """알고리즘 15.1. 키를 블록으로 나누고 블록마다 최대 m, 합 l, 누적 o 를 갱신한다.
    점수 행렬을 안 만든다.
    """
    scale = 1.0 / np.sqrt(q.shape[1])
    n = k.shape[0]
    m_run = np.full(q.shape[0], -np.inf)
    l_run = np.zeros(q.shape[0])
    o_run = np.zeros((q.shape[0], v.shape[1]))
    for s in range(0, n, block):
        e = min(s + block, n)
        sc = q @ k[s:e].T * scale                        # (m, b) 블록 점수만
        m_new = np.maximum(m_run, sc.max(axis=1))
        alpha = np.exp(m_run - m_new)                     # 옛 누적을 새 최대에 맞춘다
        p = np.exp(sc - m_new[:, None])
        l_run = alpha * l_run + p.sum(axis=1)
        o_run = alpha[:, None] * o_run + p @ v[s:e]
        m_run = m_new
    return o_run / l_run[:, None]


def traffic_naive(m: int, n: int, d: int, dv: int, itemsize: int) -> int:
    """명제 15.2. 점수 행렬 쓰기 + 읽기 + 확률 쓰기 + 읽기: 4 m n 원소.
    입력과 출력은 별도.
    """
    return itemsize * (4 * m * n + m * d + n * d + n * dv + m * dv)


def traffic_tiled(m: int, n: int, d: int, dv: int, itemsize: int, block: int) -> int:
    """타일마다 키와 값 블록을 한 번 읽고, 질의는 캐시에 머문다. 점수 행렬이 없다."""
    return itemsize * (m * d + n * d + n * dv + m * dv)
