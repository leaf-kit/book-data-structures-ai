"""15장. 캐시 축출. LRU 에서 헤비 히터로.

KV 캐시가 예산을 넘으면 토큰을 뗀다. 무엇을 떼느냐가 정책이다.
LRU 는 오래된 토큰을 떼고, 헤비 히터는 지금까지 어텐션 점수의 누적이 작은 토큰을 뗀다.
어텐션 출력이 정책마다 얼마나 어긋나는지를 정확한 어텐션과의 차이로 잰다.
"""
from __future__ import annotations

import numpy as np


def softmax_rows(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=1, keepdims=True)


def choose_victim(idx: np.ndarray, score_sum: np.ndarray, policy: str,
                  recent: int) -> int:
    """두 정책. lru 는 가장 오래된 토큰, heavy 는 최근 recent 개를 뺀 나머지에서
    누적 어텐션 점수가 가장 작은 토큰.
    """
    if policy == "lru":
        return int(idx[0])
    cand = idx[: max(len(idx) - recent, 1)]
    return int(cand[np.argmin(score_sum[cand])])


def decode_with_policy(q: np.ndarray, k: np.ndarray, v: np.ndarray, budget: int,
                       policy: str, recent: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """알고리즘 15.2. 토큰을 하나씩 디코딩하며 캐시를 예산 안에 둔다.
    q, k, v 는 (n, d). 단계 t 는 질의 q[t] 로 캐시 안의 키 (t 이하) 를 본다.
    policy: full (안 뗌) | lru | heavy. (출력 (n, dv), 단계마다 남긴 토큰 수).
    """
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    keep = np.zeros(n, dtype=bool)
    score_sum = np.zeros(n)
    out = np.zeros((n, v.shape[1]))
    kept_count = np.zeros(n, dtype=np.int64)
    for t in range(n):
        keep[t] = True
        idx = np.flatnonzero(keep)
        s = (k[idx] @ q[t]) * scale
        p = np.exp(s - s.max())
        p /= p.sum()
        out[t] = p @ v[idx]
        score_sum[idx] += p                             # 누적 점수. 헤비 히터의 근거
        kept_count[t] = len(idx)
        if len(idx) > budget:
            keep[choose_victim(idx, score_sum, policy, recent)] = False
    return out, kept_count


def full_attention_causal(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    n, d = q.shape
    s = q @ k.T / np.sqrt(d)
    s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, -np.inf)
    return softmax_rows(s) @ v
