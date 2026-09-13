"""문제 6.8. 미룬 키를 옆 자리 대신 걸음마다 새 해시로 보내는 한꺼번에 넣기."""
import numpy as np

from dsai.hashing import random_odd, universal_hash


def batch_insert_two_hash(keys: np.ndarray, m: int, a1: int, a2: int,
                          max_rounds: int = 64) -> int:
    table = np.full(m, -1, dtype=np.int64)
    pending = keys.copy()
    rounds = 0
    while len(pending) and rounds < max_rounds:
        rounds += 1
        # 걸음마다 다른 홀수 계수. 뭉친 키들이 다음 걸음에는 흩어진다
        a = ((a1 * (2 * rounds + 1) + a2 * rounds) & ((1 << 64) - 1)) | 1
        slots = universal_hash(pending, a, m)
        order = np.argsort(slots, kind="stable")
        slots, pending = slots[order], pending[order]
        first = np.concatenate(([True], slots[1:] != slots[:-1]))
        win = first & (table[slots] == -1)
        table[slots[win]] = pending[win]
        pending = pending[~win]
    return rounds if not len(pending) else -1


def test_two_hash_needs_fewer_rounds():
    from dsai.parhash import batch_insert_rounds
    rng = np.random.default_rng(2)
    m = 1 << 14
    keys = rng.choice(1 << 31, int(m * 0.5), replace=False).astype(np.int64)
    a1, a2 = random_odd(rng), random_odd(rng)
    two = batch_insert_two_hash(keys, m, a1, a2)
    one = batch_insert_rounds(keys, m, a1)
    assert 0 < two <= one
