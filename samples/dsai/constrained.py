"""9장. 제약 디코딩.
오토마톤의 상태마다 허용되는 토큰의 집합을 트라이 위에서 미리 구한다.

문법을 유한 상태 오토마톤으로 두면,
상태 s 에서 토큰 t 가 허용된다는 것은 t 의 바이트를 차례로 먹어 죽지 않는다는 것이다.
어휘 전부를 상태마다 하나씩 확인하면 |V| x |상태| x 토큰 길이 이고,
트라이를 오토마톤과 함께 내려가면 공통 접두사가 한 번만 확인된다.
"""
from __future__ import annotations

import numpy as np

from dsai.trie import NIL, Trie

DEAD = -1


class DFA:
    """정의 9.12. 상태 집합과 바이트 전이표. 전이표는 (상태 수,
    256) 배열이고 -1 이 죽음이다.
    """

    def __init__(self, n_states: int, start: int, accept: set[int]):
        self.delta = np.full((n_states, 256), DEAD, dtype=np.int64)
        self.start = start
        self.accept = accept

    def add(self, s: int, chars: bytes, t: int) -> None:
        for c in chars:
            self.delta[s, c] = t

    def step(self, s: int, data: bytes) -> int:
        for c in data:
            s = self.delta[s, c]
            if s == DEAD:
                return DEAD
        return s


def number_dfa() -> DFA:
    """예. 정수 또는 소수: [0-9]+ ( '.' [0-9]+ )?  상태 0 시작, 1 정수부, 2 점,
    3 소수부.
    """
    d = DFA(4, 0, {1, 3})
    digits = b"0123456789"
    d.add(0, digits, 1)
    d.add(1, digits, 1)
    d.add(1, b".", 2)
    d.add(2, digits, 3)
    d.add(3, digits, 3)
    return d


def allowed_by_scan(dfa: DFA, vocab: dict[int, bytes], state: int) -> set[int]:
    """디딤돌. 어휘 전부를 하나씩 상태에서 먹여 본다."""
    return {t for t, w in vocab.items() if dfa.step(state, w) != DEAD}


def allowed_by_trie(dfa: DFA, trie: Trie, state: int) -> set[int]:
    """알고리즘 9.4. 트라이와 오토마톤을 같이 내려간다. 죽는 가지는 통째로 버린다."""
    out: set[int] = set()
    stack = [(0, state)]
    while stack:
        node, s = stack.pop()
        if trie.token[node] != NIL:
            out.add(trie.token[node])
        for b, child in trie.children_of(node):
            t = dfa.delta[s, b]
            if t != DEAD:
                stack.append((child, t))
    return out


def build_mask_table(dfa: DFA, trie: Trie, vocab_size: int) -> np.ndarray:
    """상태마다 허용 토큰의 비트 마스크. (상태 수, 어휘 크기) bool 배열."""
    table = np.zeros((dfa.delta.shape[0], vocab_size), dtype=bool)
    for s in range(dfa.delta.shape[0]):
        for t in allowed_by_trie(dfa, trie, s):
            table[s, t] = True
    return table
