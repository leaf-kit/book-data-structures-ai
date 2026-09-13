"""16장. 체크포인트와 내용 주소 저장. 머클 트리.

큰 파일을 블록으로 나누고 블록마다 해시를 두면,
두 체크포인트에서 같은 블록은 한 번만 저장한다.
블록 해시를 다시 해시해 트리를 올리면 뿌리 하나가 파일 전체를 대신하고,
다른 블록을 로그 개의 해시로 찾는다.
"""
from __future__ import annotations

import hashlib

import numpy as np


def block_hashes(data: bytes, block: int) -> list[bytes]:
    """정의 16.2. 블록마다 SHA-256. 내용이 주소다."""
    return [hashlib.sha256(data[i:i + block]).digest()
            for i in range(0, len(data), block)]


def merkle_root(hashes: list[bytes]) -> tuple[bytes, list[list[bytes]]]:
    """알고리즘 16.2. 이웃 둘의 해시를 이어 다시 해시한다. 층이 로그 개다. (뿌리,
    층들).
    """
    levels = [list(hashes)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        if len(cur) % 2:
            cur = cur + [cur[-1]]
        levels.append([hashlib.sha256(cur[i] + cur[i + 1]).digest()
                       for i in range(0, len(cur), 2)])
    return levels[-1][0], levels


def changed_blocks(levels_a: list[list[bytes]],
                   levels_b: list[list[bytes]]) -> list[int]:
    """명제 16.3. 뿌리에서 내려가며 다른 부분 트리만 연다.
    다른 블록 k 개에 해시 비교 O(k log n).
    """
    depth = len(levels_a) - 1
    stack = [(depth, 0)]
    out = []
    while stack:
        lvl, i = stack.pop()
        a = levels_a[lvl][i] if i < len(levels_a[lvl]) else None
        b = levels_b[lvl][i] if i < len(levels_b[lvl]) else None
        if a == b:
            continue
        if lvl == 0:
            out.append(i)
        else:
            stack.append((lvl - 1, 2 * i + 1))
            stack.append((lvl - 1, 2 * i))
    return sorted(out)


class ContentStore:
    """내용 주소 저장소. 해시 -> 블록. 같은 블록은 한 번만 든다."""

    def __init__(self):
        self.blocks: dict[bytes, bytes] = {}
        self.stored = 0

    def put(self, data: bytes, block: int) -> list[bytes]:
        hs = block_hashes(data, block)
        for h, i in zip(hs, range(0, len(data), block)):
            if h not in self.blocks:
                self.blocks[h] = data[i:i + block]
                self.stored += len(self.blocks[h])
        return hs

    def get(self, hashes: list[bytes]) -> bytes:
        return b"".join(self.blocks[h] for h in hashes)
