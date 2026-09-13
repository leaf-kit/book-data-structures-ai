"""3장. 희소 행렬. 0 이 아닌 원소만 연속으로 놓는 세 가지 압축.

COO 는 (행, 열, 값) 세 배열이고, CSR 은 행 포인터로 행을 접은 것이고,
CSC 는 열을 접은 것이다.
같은 행렬이라도 어느 연산이 빠른지가 접은 방향에 달렸다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class COO:
    """좌표 목록. 원소마다 (행, 열, 값). 만들기 쉽고 곱하기는 느리다."""

    shape: tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    @property
    def nnz(self) -> int:
        return len(self.vals)

    def nbytes(self) -> int:
        return self.rows.nbytes + self.cols.nbytes + self.vals.nbytes


@dataclass(frozen=True)
class CSR:
    """정의 3.2. 행 포인터, 열 번호, 값. 행 i 의 원소는 vals[ptr[i]:ptr[i+1]] 이다."""

    shape: tuple[int, int]
    ptr: np.ndarray    # 길이 m + 1
    cols: np.ndarray   # 길이 nnz
    vals: np.ndarray   # 길이 nnz

    @property
    def nnz(self) -> int:
        return len(self.vals)

    def nbytes(self) -> int:
        return self.ptr.nbytes + self.cols.nbytes + self.vals.nbytes

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """행 하나. 불변식 덕에 자르기 하나로 끝난다."""
        a, b = self.ptr[i], self.ptr[i + 1]
        return self.cols[a:b], self.vals[a:b]


def dense_to_coo(a: np.ndarray) -> COO:
    rows, cols = np.nonzero(a)
    return COO(a.shape, rows.astype(np.int32), cols.astype(np.int32), a[rows, cols])


def coo_to_csr(c: COO) -> CSR:
    """알고리즘 3.1. 행 번호로 정렬한 뒤 행마다 시작 위치를 센다."""
    order = np.lexsort((c.cols, c.rows))
    rows, cols, vals = c.rows[order], c.cols[order], c.vals[order]
    counts = np.bincount(rows, minlength=c.shape[0])
    ptr = np.zeros(c.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return CSR(c.shape, ptr, cols, vals)


def dense_to_csr(a: np.ndarray) -> CSR:
    return coo_to_csr(dense_to_coo(a))


def csr_matvec(m: CSR, x: np.ndarray) -> np.ndarray:
    """정리 3.1. 0 이 아닌 원소마다 곱 하나 합 하나. 이동량은 nnz 에 비례한다."""
    prod = m.vals * x[m.cols]
    out = np.zeros(m.shape[0], dtype=prod.dtype)
    nonempty = m.ptr[:-1] != m.ptr[1:]
    sums = np.add.reduceat(prod, m.ptr[:-1][nonempty]) if nonempty.any() else prod[:0]
    out[nonempty] = sums
    return out


def csr_transpose(m: CSR) -> CSR:
    """CSR 의 전치는 CSC 다. 열을 접은 것이므로 열 접근이 빠르다."""
    coo = COO((m.shape[1], m.shape[0]), m.cols, expand_ptr(m.ptr), m.vals)
    return coo_to_csr(coo)


def expand_ptr(ptr: np.ndarray) -> np.ndarray:
    """행 포인터를 행 번호 배열로 되돌린다. COO 로 가는 길이다."""
    counts = np.diff(ptr)
    return np.repeat(np.arange(len(counts), dtype=np.int32), counts)


def random_sparse(m: int, n: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, n), dtype=np.float32)
    mask = rng.random((m, n)) < density
    return a * mask
