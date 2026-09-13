"""5장. 스택. 마지막에 넣은 것이 먼저 나오는 배열.

호출 스택은 언어가 대신 관리하는 스택이다. 깊이에 상한이 있고,
프레임 하나가 값 하나보다 훨씬 크다.
같은 일을 명시적 스택으로 바꾸면 상한이 사라지고 프레임이 정수 하나가 된다.
"""
from __future__ import annotations

import numpy as np


class Stack:
    """정의 5.1. 배열과 꼭대기 번호. push 는 끝에 쓰기, pop 은 끝에서 읽기."""

    def __init__(self, capacity: int, dtype=np.int64):
        self.buf = np.empty(capacity, dtype=dtype)
        self.top = 0

    def push(self, v) -> None:
        self.buf[self.top] = v
        self.top += 1

    def pop(self):
        self.top -= 1
        return self.buf[self.top]

    def __len__(self) -> int:
        return self.top


def chain_children(n: int) -> list[list[int]]:
    """노드 i 의 자식이 i + 1 하나뿐인 트리. 깊이가 n 이다."""
    return [[i + 1] if i + 1 < n else [] for i in range(n)]


def depth_recursive(children: list[list[int]], node: int = 0) -> int:
    """재귀. 프레임이 호출 스택에 쌓인다. 깊이에 상한이 있다."""
    best = 0
    for c in children[node]:
        best = max(best, depth_recursive(children, c))
    return best + 1


def depth_iterative(children: list[list[int]]) -> int:
    """알고리즘 5.1. 명시적 스택. 프레임이 (노드, 깊이) 정수 둘이다."""
    stack = [(0, 1)]
    best = 0
    while stack:
        node, d = stack.pop()
        best = max(best, d)
        for c in children[node]:
            stack.append((c, d + 1))
    return best
