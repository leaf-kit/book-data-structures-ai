"""문제 7.5. bool 배열을 진짜 비트로 접는다.
해시가 같으므로 오답률이 같고 메모리가 8 분의 1 이다.
"""
import numpy as np

from dsai.bloom import BloomFilter, hashes, optimal_k


class PackedBloom(BloomFilter):
    def __init__(self, m_bits: int, k: int):
        self.m = m_bits
        self.k = k
        self.bits = np.zeros((m_bits + 7) // 8, dtype=np.uint8)

    def add(self, keys: np.ndarray) -> None:
        pos = hashes(keys, self.k, self.m).ravel()
        np.bitwise_or.at(self.bits, pos >> 3, (1 << (pos & 7)).astype(np.uint8))

    def contains(self, keys: np.ndarray) -> np.ndarray:
        pos = hashes(keys, self.k, self.m)
        hit = (self.bits[pos >> 3] >> (pos & 7).astype(np.uint8)) & 1
        return hit.all(axis=1)

    def nbytes(self) -> int:
        return self.bits.nbytes


def test_packed_matches_bool():
    rng = np.random.default_rng(0)
    keys = rng.choice(1 << 40, 40_000, replace=False).astype(np.int64)
    inside, outside = keys[:20_000], keys[20_000:]
    m, k = 20_000 * 10, optimal_k(20_000 * 10, 20_000)
    a, b = BloomFilter(m, k), PackedBloom(m, k)
    a.add(inside)
    b.add(inside)
    assert b.contains(inside).all()
    assert np.array_equal(a.contains(outside), b.contains(outside))  # 같은 답
    assert b.nbytes() == a.bits.nbytes // 8      # 메모리 8 분의 1
