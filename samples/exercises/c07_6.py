"""문제 7.6. 블록 블룸 필터. 자리 k 개를 512 비트 블록 하나에 모은다.
라인 하나에 오답률은 조금 오른다.
"""
import numpy as np

from dsai.bloom import BloomFilter, hashes, optimal_k
from dsai.hashing import mix64

BLOCK = 512


class BlockedBloom(BloomFilter):
    def _positions(self, keys: np.ndarray) -> np.ndarray:
        nblocks = self.m // BLOCK
        h = mix64(keys.astype(np.uint64) ^ np.uint64(0xB10C))
        block = (h % np.uint64(nblocks)).astype(np.int64)
        inner = hashes(keys, self.k, BLOCK)                # 블록 안의 자리 k 개
        return block[:, None] * BLOCK + inner

    def add(self, keys: np.ndarray) -> None:
        self.bits[self._positions(keys).ravel()] = True

    def contains(self, keys: np.ndarray) -> np.ndarray:
        return self.bits[self._positions(keys)].all(axis=1)


def fpr(filt, inside, outside):
    filt.add(inside)
    assert filt.contains(inside).all()
    return filt.contains(outside).mean()


def test_blocked_is_one_line_and_slightly_worse():
    rng = np.random.default_rng(1)
    n = 200_000
    keys = rng.choice(1 << 40, 2 * n, replace=False).astype(np.int64)
    inside, outside = keys[:n], keys[n:]
    m, k = 16 * n, optimal_k(16 * n, n)
    plain = fpr(BloomFilter(m, k), inside, outside)
    blocked = fpr(BlockedBloom(m, k), inside, outside)
    pos = BlockedBloom(m, k)._positions(inside[:100])
    assert ((pos // BLOCK) == (pos[:, :1] // BLOCK)).all()   # 자리 k 개가 한 블록
    assert plain <= blocked < 4 * plain + 1e-3    # 조금 나쁘고 크게 나쁘지 않다
