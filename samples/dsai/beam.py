"""10장. 빔 탐색의 힙. 후보 B 개에서 어휘 V 개로 뻗은 B x V 중 상위 B 개를 고른다.

한 단계는 (B, V) 점수 행렬에서 top-B 를 고르는 것이고, 그것이 10.1 절의 선택이다.
이어진 열은 부모 번호 배열로 적는다. 5장의 명시적 스택 대신 되짚기 배열이다.
"""
from __future__ import annotations

import numpy as np


def beam_step(scores: np.ndarray, logp: np.ndarray,
              beam: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """알고리즘 10.2. scores (B,) 와 logp (B, V) 에서 상위 beam 개의 (부모, 토큰,
    점수).
    """
    total = scores[:, None] + logp                   # (B, V)
    flat = total.ravel()
    idx = np.argpartition(-flat, beam - 1)[:beam]    # 선택. 정렬하지 않는다
    parent, token = np.divmod(idx, logp.shape[1])
    return parent, token, flat[idx]


def beam_search(step_logp, beam: int, steps: int,
                vocab: int) -> tuple[np.ndarray, float]:
    """빔 탐색 전체. step_logp(prev_tokens (B,), t) -> (B, V) 로그 확률.
    최선 열과 점수.
    """
    scores = np.zeros(1)
    tokens = np.zeros(1, dtype=np.int64)
    parents: list[np.ndarray] = []
    chosen: list[np.ndarray] = []
    for t in range(steps):
        logp = step_logp(tokens, t)
        parent, token, scores = beam_step(scores, logp, min(beam, logp.size))
        parents.append(parent)
        chosen.append(token)
        tokens = token
    best = int(np.argmax(scores))
    seq = []
    for t in range(steps - 1, -1, -1):              # 부모 배열을 거슬러 올라간다
        seq.append(int(chosen[t][best]))
        best = int(parents[t][best])
    return np.array(seq[::-1]), float(scores.max())


def exact_best(table: np.ndarray) -> float:
    """디딤돌. 조건부 표 (steps, V, V) 위의 최선 점수를 동적 계획법으로.
    열을 전부 보는 것과 같은 답.
    """
    steps, vocab, _ = table.shape
    best = table[0][0].copy()                       # 첫 토큰. 이전 토큰은 0 으로 둔다
    for t in range(1, steps):
        best = (best[:, None] + table[t]).max(axis=0)
    return float(best.max())
