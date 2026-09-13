"""2장. 할당자 시뮬레이션. 연속 할당의 외부 단편화와 페이지 할당의 내부 단편화.

요청이 서로 다른 길이로 들어오고 나가면,
연속 할당은 빈틈이 생겨 총합은 남는데 자리가 없는 상태가 된다.
페이지 할당은 빈틈 대신 마지막 페이지의 자투리만 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContiguousArena:
    """첫 맞춤 (first fit) 연속 할당. 정의 2.2 의 외부 단편화를 만든다."""

    total: int
    free: list[tuple[int, int]] = field(default_factory=list)  # (start, length)
    used: dict[int, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.free = [(0, self.total)]

    def alloc(self, key: int, length: int) -> bool:
        for i, (start, size) in enumerate(self.free):
            if size >= length:
                self.used[key] = (start, length)
                if size == length:
                    self.free.pop(i)
                else:
                    self.free[i] = (start + length, size - length)
                return True
        return False

    def release(self, key: int) -> None:
        start, length = self.used.pop(key)
        self.free.append((start, length))
        self.free.sort()
        merged: list[tuple[int, int]] = []
        for s, ln in self.free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + ln)
            else:
                merged.append((s, ln))
        self.free = merged

    def free_total(self) -> int:
        return sum(ln for _, ln in self.free)

    def largest_free(self) -> int:
        return max((ln for _, ln in self.free), default=0)

    def external_fragmentation(self) -> float:
        """빈 공간 중 가장 큰 빈 구간에 못 드는 비율. 0 이면 빈틈이 없다."""
        f = self.free_total()
        return 0.0 if f == 0 else 1.0 - self.largest_free() / f


@dataclass
class PagedArena:
    """고정 크기 페이지 할당. 외부 단편화가 없고 내부 단편화만 남는다."""

    total: int
    page: int
    free_pages: list[int] = field(default_factory=list)
    used: dict[int, list[int]] = field(default_factory=dict)
    lengths: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.free_pages = list(range(self.total // self.page))

    def pages_for(self, length: int) -> int:
        return -(-length // self.page)

    def alloc(self, key: int, length: int) -> bool:
        need = self.pages_for(length)
        if need > len(self.free_pages):
            return False
        self.used[key] = [self.free_pages.pop() for _ in range(need)]
        self.lengths[key] = length
        return True

    def release(self, key: int) -> None:
        self.free_pages.extend(self.used.pop(key))
        del self.lengths[key]

    def free_total(self) -> int:
        return len(self.free_pages) * self.page

    def internal_fragmentation(self) -> float:
        """할당된 페이지 중 실제로 안 쓰는 자투리의 비율."""
        alloc = sum(len(p) for p in self.used.values()) * self.page
        used = sum(self.lengths.values())
        return 0.0 if alloc == 0 else 1.0 - used / alloc
