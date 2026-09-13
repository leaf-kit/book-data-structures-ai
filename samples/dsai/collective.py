"""16장. 샤딩과 집합 통신. 링 위의 all-reduce.

기계 p 대가 각자 기울기 벡터를 갖고, 전부의 합을 전부가 가져야 한다.
하나가 다 모아 나눠 주면 그 하나의 링크에 (p-1) 벡터가 몰린다.
링은 벡터를 p 조각으로 잘라 조각을 돌린다. 기계마다 보내는 양이 2 (p-1) / p 벡터라
p 와 거의 무관하다. 이 모듈은 링크마다 보낸 바이트를 센다.
"""
from __future__ import annotations

import numpy as np


def allreduce_star(shards: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """디딤돌. 기계 0 이 전부 받아 더하고 전부에게 보낸다. (합, 기계 0 의 링크 바이트,
    단계 수).
    """
    p = len(shards)
    total = shards[0].copy()
    bytes_on_root = 0
    for s in shards[1:]:
        total += s
        bytes_on_root += s.nbytes
    bytes_on_root += (p - 1) * total.nbytes
    return total, bytes_on_root, 2


def ring_step(buf: list[np.ndarray], chunks: list[np.ndarray], t: int, shift: int,
              add: bool) -> int:
    """단계 t. 기계 i 가 조각 (i + shift - t) 를 오른쪽 이웃에게 보낸다.
    add 면 이웃이 더하고, 아니면 덮어쓴다. 링크 하나가 나른 바이트를 돌려준다.
    """
    p = len(buf)
    sends = [(i, chunks[(i + shift - t) % p]) for i in range(p)]
    data = [buf[i][c].copy() for i, c in sends]           # 동시에 보내므로 먼저 복사
    for (i, c), d in zip(sends, data):
        if add:
            buf[(i + 1) % p][c] += d
        else:
            buf[(i + 1) % p][c] = d
    return max(d.nbytes for d in data)


def allreduce_ring(shards: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """알고리즘 16.1. reduce-scatter 와 all-gather. 조각 p 개를 링으로 돌린다.
    (합, 링크 하나의 바이트, 단계 수).
    """
    p, n = len(shards), len(shards[0])
    chunks = np.array_split(np.arange(n), p)
    buf = [s.copy() for s in shards]
    per_link = 0
    for t in range(p - 1):                                 # reduce-scatter: 더한다
        per_link = max(per_link, ring_step(buf, chunks, t, 0, True))
    for t in range(p - 1):                                 # all-gather: 덮어쓴다
        ring_step(buf, chunks, t, 1, False)
    return buf[0], per_link * 2 * (p - 1), 2 * (p - 1)


def bytes_per_machine(n: int, itemsize: int, p: int, algo: str) -> int:
    """명제 16.1. 기계 하나가 보내는 바이트. 별은 뿌리가 2 (p-1) n,
    링은 모두가 2 (p-1) n / p.
    """
    if algo == "star":
        return 2 * (p - 1) * n * itemsize
    return 2 * (p - 1) * n * itemsize // p
