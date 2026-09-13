"""6장. 병렬 삽입. 키 수백만 개를 한꺼번에 넣을 때의 두 길.

하나씩 넣는 해시 테이블은 삽입마다 자리를 보고 쓴다.
한꺼번에 넣으면 같은 자리를 둘이 쓰려는 충돌이 난다.
정렬 기반은 키를 정렬해 같은 키를 붙여 놓고 한 번에 센다.
충돌이 없고 전부가 순차 접근이다.
"""
from __future__ import annotations

import numpy as np

from dsai.hashing import universal_hash


def count_serial(keys: np.ndarray) -> dict[int, int]:
    """하나씩. 파이썬 dict 삽입 n 번."""
    counts: dict[int, int] = {}
    for k in keys.tolist():
        counts[k] = counts.get(k, 0) + 1
    return counts


def count_sorted(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """알고리즘 6.2. 정렬하고 경계를 찾아 센다. 해시 테이블 없이 같은 답."""
    s = np.sort(keys)
    change = np.concatenate(([True], s[1:] != s[:-1]))
    uniq = s[change]
    starts = np.nonzero(change)[0]
    counts = np.diff(np.concatenate((starts, [len(s)])))
    return uniq, counts


def batch_insert_conflicts(keys: np.ndarray, m: int, a: int) -> tuple[int, int]:
    """한 걸음에 키 전부를 자기 자리에 쓰려 할 때,
    자리가 겹친 키의 수와 실제로 쓰인 키의 수.
    """
    slots = universal_hash(keys, a, m)
    uniq, counts = count_sorted(slots)
    conflicted = int(counts[counts > 1].sum() - len(counts[counts > 1]))
    return conflicted, len(keys) - conflicted


def batch_insert_rounds(keys: np.ndarray, m: int, a: int) -> int:
    """알고리즘 6.3. 겹친 키를 다음 걸음으로 미루며 몇 걸음에 다 들어가는지."""
    table = np.full(m, -1, dtype=np.int64)
    pending = keys.copy()
    rounds = 0
    offset = 0
    while len(pending):
        rounds += 1
        slots = (universal_hash(pending, a, m) + offset) % m
        order = np.argsort(slots, kind="stable")
        slots, pending = slots[order], pending[order]
        first = np.concatenate(([True], slots[1:] != slots[:-1]))
        free = table[slots] == -1
        win = first & free
        table[slots[win]] = pending[win]
        pending = pending[~win]
        offset += 1
    return rounds
