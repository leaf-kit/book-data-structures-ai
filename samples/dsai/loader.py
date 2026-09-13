"""16장. 셔플과 순열. 데이터 로더의 자료 구조.

학습은 매 에폭 데이터를 무작위 순서로 본다. 순열 하나가 그 순서다.
순열을 만드는 것이 Fisher-Yates 이고,
순열대로 읽으면 무작위 접근이다.
블록 단위로 섞으면 블록 안은 순차이고 블록 사이만 무작위다.
큰 데이터는 순열을 저장하지 않고 위치에서 계산한다. 그것이 형식 보존 순열이다.
"""
from __future__ import annotations

import numpy as np


def fisher_yates(n: int, rng: np.random.Generator) -> np.ndarray:
    """알고리즘 16.3. 뒤에서부터 무작위 자리와 바꾼다. 균등한 순열, O(n)."""
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        p[i], p[j] = p[j], p[i]
    return p


def block_shuffle(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """정의 16.5. 블록 순서를 섞고 블록 안도 섞는다. 블록 사이만 무작위 접근이다."""
    nb = (n + block - 1) // block
    order = rng.permutation(nb)
    out = np.empty(n, dtype=np.int64)
    k = 0
    for b in order:
        lo, hi = b * block, min((b + 1) * block, n)
        inner = rng.permutation(hi - lo) + lo
        out[k:k + hi - lo] = inner
        k += hi - lo
    return out


def feistel_round(y: np.ndarray, h: int, keys: np.ndarray, rounds: int) -> np.ndarray:
    """2h 비트 위의 Feistel 망. 반쪽 h 비트씩. 어떤 f 를 써도 전단사다."""
    mask = (1 << h) - 1
    left, right = y >> h, y & mask
    for r in range(rounds):
        f = ((right * 0x9E3779B1 + int(keys[r])) * 0x85EBCA6B) & 0xFFFFFFFF
        left, right = right, left ^ (f & mask)
    return (left << h) | right


def feistel_permutation(i: np.ndarray, n: int, keys: np.ndarray,
                        rounds: int = 4) -> np.ndarray:
    """알고리즘 16.4. 저장하지 않는 순열. 2^{2h} >= n 인 h 로 망을 돌리고,
    n 을 넘으면 넘지 않을 때까지 다시 돌린다 (사이클 걷기).
    """
    h = (max(1, int(np.ceil(np.log2(n)))) + 1) // 2
    out = i.astype(np.int64).copy()
    pending = np.ones(len(out), dtype=bool)
    for _ in range(64):
        y = feistel_round(out, h, keys, rounds)
        out = np.where(pending, y, out)
        pending &= out >= n
        if not pending.any():
            break
    return out


def sequential_reads(order: np.ndarray, block: int) -> int:
    """읽는 순서에서 같은 블록 안에 연속으로 머문 횟수. 높을수록 순차 읽기가 많다."""
    b = order // block
    return int(np.sum(b[1:] == b[:-1]))
