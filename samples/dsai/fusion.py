"""11장. 연산자 융합. 그래프를 다시 쓰다.

원소별 연산의 사슬 y = f_k(...f_1(x)) 를 하나씩 하면 연산마다 배열 전체를 읽고 쓴다.
융합은 중간 결과를 메모리에 안 내보내는 것이다.
NumPy 에서는 캐시에 드는 블록 단위로 사슬 전체를 도는 것으로 흉내 낸다.
"""
from __future__ import annotations

import numpy as np

# 원소별 연산의 사슬. 각각이 ufunc 하나라 배열 전체를 한 번 읽고 한 번 쓴다.
CHAIN_CHEAP = [lambda t: np.multiply(t, 1.5), lambda t: np.add(t, 0.5), np.square,
               np.negative, np.abs]
CHAIN_HEAVY = [np.tanh, np.exp, np.sqrt, np.log1p, np.sin]
CHAIN = CHAIN_CHEAP


def unfused(x: np.ndarray, chain=CHAIN) -> np.ndarray:
    """디딤돌. 연산 k 개를 차례로. 중간 결과 k-1 개가 메모리를 오간다."""
    y = x
    for f in chain:
        y = f(y)
    return y


def fused_blocks(x: np.ndarray, out: np.ndarray, block: int, chain=CHAIN) -> np.ndarray:
    """알고리즘 11.5. 블록마다 사슬 전체를 돈다. 중간 결과가 캐시 안에 머문다."""
    n = len(x)
    for s in range(0, n, block):
        e = min(s + block, n)
        y = x[s:e]
        for f in chain:
            y = f(y)
        out[s:e] = y
    return out


def traffic_bytes(n: int, k: int, itemsize: int, fused: bool) -> int:
    """명제 11.8. 연산 k 개 사슬의 메모리 이동량. 융합 전 2k n,
    융합 후 2 n (바이트 단위).
    """
    return (2 * n if fused else 2 * k * n) * itemsize


def fusable_chains(op: np.ndarray, indeg: np.ndarray, outdeg: np.ndarray,
                   elementwise: set[int]) -> list[list[int]]:
    """정의 11.7. 융합 가능한 사슬 찾기.
    원소별 연산이고 출력을 하나만 쓰는 노드를 잇는다.
    """
    chains, cur = [], []
    for i in range(len(op)):
        if op[i] in elementwise and indeg[i] <= 1 and (not cur or outdeg[cur[-1]] == 1):
            cur.append(i)
        else:
            if len(cur) > 1:
                chains.append(cur)
            cur = [i] if op[i] in elementwise else []
    if len(cur) > 1:
        chains.append(cur)
    return chains
