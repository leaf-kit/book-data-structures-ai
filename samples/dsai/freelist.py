"""4장. 자유 리스트. 같은 크기의 칸을 상수 시간에 주고 받는 할당자.

빈 칸끼리 연결 리스트로 이어 둔다. 주는 것은 머리를 떼는 것,
받는 것은 머리에 붙이는 것이다.
연결 리스트가 지금도 살아 있는 자리가 여기다. 순회하지 않고 머리만 쓴다.
"""
from __future__ import annotations

import numpy as np


class FreeList:
    """정의 4.2. 칸 크기가 같은 풀.
    빈 칸은 자기 자리에 다음 빈 칸의 번호를 적어 둔다.
    """

    def __init__(self, capacity: int):
        self.nxt = np.arange(1, capacity + 1, dtype=np.int64)
        self.nxt[-1] = -1
        self.head = 0
        self.in_use = 0

    def alloc(self) -> int:
        """머리를 뗀다. 읽기 하나, 쓰기 하나."""
        i = self.head
        if i < 0:
            raise MemoryError("빈 칸이 없다")
        self.head = self.nxt[i]
        self.in_use += 1
        return i

    def free(self, i: int) -> None:
        """머리에 붙인다. 최근에 놓은 칸이 다음에 먼저 나간다.
        캐시에 남아 있을 확률이 높다.
        """
        self.nxt[i] = self.head
        self.head = i
        self.in_use -= 1


class SlabPool:
    """크기 등급마다 자유 리스트 하나.
    요청 크기를 등급으로 올림해 그 리스트에서 준다.
    """

    def __init__(self, sizes: list[int], per_class: int):
        self.sizes = sorted(sizes)
        self.lists = {s: FreeList(per_class) for s in self.sizes}

    def size_class(self, length: int) -> int:
        for s in self.sizes:
            if length <= s:
                return s
        raise ValueError(f"등급보다 큰 요청 {length}")

    def alloc(self, length: int) -> tuple[int, int]:
        s = self.size_class(length)
        return s, self.lists[s].alloc()

    def free(self, s: int, i: int) -> None:
        self.lists[s].free(i)

    def internal_waste(self, lengths: np.ndarray) -> float:
        """등급으로 올림한 만큼의 내부 단편화 비율."""
        classes = np.array([self.size_class(int(x)) for x in lengths])
        return float(1.0 - lengths.sum() / classes.sum())
