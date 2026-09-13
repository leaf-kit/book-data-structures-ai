"""2장. KV 캐시. 요청마다 자라는 추가 전용 배열과 그것을 페이지로 나눈 것.

생성 모델은 토큰을 하나 낼 때마다 그 토큰의 키와 값을 저장하고,
다음 토큰을 낼 때 지금까지의 전부를 읽는다.
길이를 미리 모르는 추가 전용 배열이 요청 수만큼 있는 셈이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bytes_per_token(layers: int, heads: int, head_dim: int, elem_bytes: int) -> int:
    """정의 2.3. 토큰 하나가 캐시에 남기는 바이트. 키와 값이라 2 를 곱한다."""
    return 2 * layers * heads * head_dim * elem_bytes


@dataclass
class ContiguousKV:
    """요청마다 최대 길이만큼 미리 잡는 방식. 남는 자리가 전부 낭비다."""

    max_len: int
    dim: int
    buf: np.ndarray = field(init=False)
    length: int = 0

    def __post_init__(self) -> None:
        self.buf = np.zeros((self.max_len, self.dim), dtype=np.float32)

    def append(self, kv: np.ndarray) -> None:
        self.buf[self.length] = kv
        self.length += 1

    def reserved(self) -> int:
        return self.max_len

    def gather(self) -> np.ndarray:
        return self.buf[:self.length]


@dataclass
class BlockPool:
    """물리 블록의 풀. 블록 하나는 토큰 block_size 개의 키와 값을 담는다."""

    num_blocks: int
    block_size: int
    dim: int
    store: np.ndarray = field(init=False)
    free: list[int] = field(init=False)
    refcount: list[int] = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.num_blocks, self.block_size, self.dim)
        self.store = np.zeros(shape, dtype=np.float32)
        self.free = list(range(self.num_blocks))
        self.refcount = [0] * self.num_blocks

    def take(self) -> int:
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def share(self, b: int) -> None:
        self.refcount[b] += 1

    def drop(self, b: int) -> None:
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)

    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free)


@dataclass
class PagedKV:
    """정의 2.4. 논리 블록 번호에서 물리 블록 번호로 가는 블록 테이블을 든 KV 캐시."""

    pool: BlockPool
    table: list[int] = field(default_factory=list)
    length: int = 0

    def append(self, kv: np.ndarray) -> None:
        """알고리즘 2.2. 마지막 블록이 찼을 때만 새 블록을 받는다."""
        bs = self.pool.block_size
        if self.length % bs == 0:
            self.table.append(self.pool.take())
        block = self.table[-1]
        if self.pool.refcount[block] > 1:
            block = self._copy_on_write(len(self.table) - 1)
        self.pool.store[block, self.length % bs] = kv
        self.length += 1

    def _copy_on_write(self, logical: int) -> int:
        """공유 중인 블록에 쓰려면 먼저 내 것으로 복사한다."""
        old = self.table[logical]
        new = self.pool.take()
        self.pool.store[new] = self.pool.store[old]
        self.pool.drop(old)
        self.table[logical] = new
        return new

    def fork(self) -> "PagedKV":
        """같은 접두사를 가진 새 요청. 블록을 복사하지 않고 참조만 늘린다."""
        for b in self.table:
            self.pool.share(b)
        return PagedKV(self.pool, list(self.table), self.length)

    def release(self) -> None:
        for b in self.table:
            self.pool.drop(b)
        self.table = []
        self.length = 0

    def reserved(self) -> int:
        return len(self.table) * self.pool.block_size

    def gather(self) -> np.ndarray:
        """블록 테이블을 따라 키와 값을 논리 순서로 모은다. 어텐션이 읽는 순서다."""
        bs = self.pool.block_size
        if not self.table:
            return np.empty((0, self.pool.dim), dtype=np.float32)
        full = self.pool.store[self.table].reshape(-1, self.pool.dim)
        return full[:self.length] if bs else full
