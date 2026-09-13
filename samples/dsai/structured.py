"""3장. 블록 희소와 구조적 희소성. 0 의 자리를 미리 정해 두면 인덱스가 싸진다.

CSR 은 0 이 어디 있든 받아 주는 대신 원소마다 열 번호를 든다.
블록 희소는 블록 단위로 인덱스를 들고,
2:4 희소는 네 칸마다 두 칸이라는 약속으로 인덱스를 2비트로 줄인다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BSR:
    """정의 3.3. b x b 블록 단위의 CSR. 블록 안은 밀집이고 블록 사이만 희소다."""

    shape: tuple[int, int]
    block: int
    ptr: np.ndarray      # 블록 행마다 시작 위치
    cols: np.ndarray     # 블록 열 번호
    vals: np.ndarray     # (블록 수, b, b)

    def nbytes(self) -> int:
        return self.ptr.nbytes + self.cols.nbytes + self.vals.nbytes


def dense_to_bsr(a: np.ndarray, block: int) -> BSR:
    m, n = a.shape
    mb, nb = m // block, n // block
    tiles = a.reshape(mb, block, nb, block).transpose(0, 2, 1, 3)
    keep = np.any(tiles != 0, axis=(2, 3))
    rows, cols = np.nonzero(keep)
    ptr = np.zeros(mb + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=mb), out=ptr[1:])
    return BSR(a.shape, block, ptr, cols.astype(np.int32), tiles[rows, cols].copy())


def bsr_matvec(m: BSR, x: np.ndarray) -> np.ndarray:
    """블록마다 작은 밀집 곱. 블록 하나가 라인 여러 개를 통째로 쓴다."""
    b = m.block
    xb = x.reshape(-1, b)
    partial = np.einsum("kij,kj->ki", m.vals, xb[m.cols])
    out = np.zeros((m.shape[0] // b, b), dtype=partial.dtype)
    nonempty = m.ptr[:-1] != m.ptr[1:]
    if nonempty.any():
        out[nonempty] = np.add.reduceat(partial, m.ptr[:-1][nonempty], axis=0)
    return out.reshape(-1)


def prune_2of4(a: np.ndarray) -> np.ndarray:
    """정의 3.4. 네 칸마다 절댓값이 큰 둘만 남긴다. 나머지는 0."""
    m, n = a.shape
    groups = a.reshape(m, n // 4, 4)
    order = np.argsort(-np.abs(groups), axis=2)
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, order[:, :, :2], True, axis=2)
    return (groups * mask).reshape(m, n)


def pack_2of4(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """값 두 개와 2비트 인덱스 두 개를 그룹마다 담는다.
    인덱스는 바이트 하나에 둘을 넣는다.
    """
    m, n = a.shape
    groups = a.reshape(m, n // 4, 4)
    idx = np.argsort(groups == 0, axis=2, kind="stable")[:, :, :2]   # 0 이 아닌 자리 둘
    idx = np.sort(idx, axis=2)
    vals = np.take_along_axis(groups, idx, axis=2).astype(np.float16)
    meta = (idx[:, :, 0] | (idx[:, :, 1] << 2)).astype(np.uint8)
    return vals, meta


def matvec_2of4(vals: np.ndarray, meta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """압축된 채로 곱한다. 그룹마다 x 에서 두 원소를 골라 온다."""
    m, g, _ = vals.shape
    i0 = (meta & 3).astype(np.int64)
    i1 = (meta >> 2).astype(np.int64)
    base = np.arange(g, dtype=np.int64) * 4
    v0 = vals[:, :, 0].astype(np.float32)
    v1 = vals[:, :, 1].astype(np.float32)
    xg = x[base + i0] * v0 + x[base + i1] * v1
    return xg.sum(axis=1)


def bytes_2of4(m: int, n: int, elem_bytes: int) -> int:
    """값 절반에 그룹당 메타 1바이트."""
    return m * n // 2 * elem_bytes + m * n // 4
