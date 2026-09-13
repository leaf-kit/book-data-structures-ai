"""9장. 트라이와 최장 일치. 문자열 키의 비교를 글자 하나씩으로 나눈다.

노드를 아레나에 두고 자식은 (노드, 바이트) -> 노드 의 dict 하나로 둔다.
노드마다 dict 을 두는 것보다 메모리가 작다.
어휘를 넣으면 최장 일치 토큰화가 트라이를 한 번 내려가는 것으로 끝난다.
"""
from __future__ import annotations

import numpy as np

NIL = -1


class Trie:
    """정의 9.1. 접두사를 나누는 트리. 뿌리에서 노드까지의 경로가 문자열이다."""

    def __init__(self):
        self.child: dict[tuple[int, int], int] = {}
        self.edges: list[list[tuple[int, int]]] = [[]]   # 노드마다 (바이트, 자식) 목록
        self.token = [NIL]              # 노드가 어휘 항목의 끝이면 그 번호
        self.n = 1

    def insert(self, word: bytes, token_id: int) -> None:
        node = 0
        for b in word:
            nxt = self.child.get((node, b))
            if nxt is None:
                nxt = self.n
                self.child[(node, b)] = nxt
                self.edges[node].append((b, nxt))
                self.edges.append([])
                self.token.append(NIL)
                self.n += 1
            node = nxt
        self.token[node] = token_id

    def longest_match(self, text: bytes, start: int) -> tuple[int, int]:
        """알고리즘 9.1. start 에서 시작하는 가장 긴 어휘 항목의 (번호, 길이).
        없으면 (-1, 0).
        """
        node, best, best_len = 0, NIL, 0
        for i in range(start, len(text)):
            node = self.child.get((node, text[i]), NIL)
            if node == NIL:
                break
            if self.token[node] != NIL:
                best, best_len = self.token[node], i - start + 1
        return best, best_len

    def tokenize(self, text: bytes) -> list[int]:
        """최장 일치를 되풀이한다. 어휘에 바이트 256 개가 다 있으면 언제나 끝난다."""
        out, i = [], 0
        while i < len(text):
            tok, ln = self.longest_match(text, i)
            out.append(tok)
            i += ln
        return out

    def children_of(self, node: int) -> list[tuple[int, int]]:
        return self.edges[node]

    def nbytes(self) -> int:
        return len(self.child) * 3 * 8 + self.n * 8


def tokenize_by_dict(vocab: dict[bytes, int], text: bytes, max_len: int) -> list[int]:
    """디딤돌. 가장 긴 길이부터 줄여 가며 dict 에 물어보는 최장 일치.
    위치마다 max_len 번 해시.
    """
    out, i = [], 0
    while i < len(text):
        for ln in range(min(max_len, len(text) - i), 0, -1):
            tok = vocab.get(text[i:i + ln])
            if tok is not None:
                out.append(tok)
                i += ln
                break
    return out


def byte_vocab() -> dict[bytes, int]:
    return {bytes([b]): b for b in range(256)}
