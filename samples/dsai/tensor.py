"""1장. 배열에서 텐서로. 주소 계산, 스트라이드, 뷰, 연속성.

텐서는 평평한 버퍼 하나와 그 위를 읽는 규칙 (모양, 스트라이드, 시작 위치) 의 짝이다.
전치와 자르기는 버퍼를 건드리지 않고 규칙만 바꾼다. 그래서 값이 상수다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def offset(index: Sequence[int], strides: Sequence[int]) -> int:
    """정의 1.2 의 주소 계산. 시작 위치에서 몇 바이트 떨어져 있는가."""
    return sum(i * s for i, s in zip(index, strides))


def row_major_strides(shape: Sequence[int], itemsize: int) -> tuple[int, ...]:
    """마지막 축이 가장 촘촘하다. C 와 NumPy 의 기본이다."""
    strides = []
    acc = itemsize
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))


def col_major_strides(shape: Sequence[int], itemsize: int) -> tuple[int, ...]:
    """첫 축이 가장 촘촘하다. Fortran 과 BLAS 의 기본이다."""
    strides = []
    acc = itemsize
    for dim in shape:
        strides.append(acc)
        acc *= dim
    return tuple(strides)


def is_contiguous(shape: Sequence[int], strides: Sequence[int], itemsize: int) -> bool:
    """행 우선으로 연속인가. 연속이면 순회 한 번의 이동량이 최소다."""
    return tuple(strides) == row_major_strides(shape, itemsize)


def elements_per_line(stride_bytes: int, line_bytes: int) -> float:
    """정리 1.1. 라인 하나를 가져와서 쓰는 원소 수. 간격이 라인보다 크면 하나다."""
    if stride_bytes >= line_bytes:
        return 1.0
    return line_bytes / stride_bytes


def lines_for_traversal(n: int, stride_bytes: int, line_bytes: int) -> int:
    """원소 n 개를 간격 stride_bytes 로 훑을 때 옮겨지는 라인 수."""
    return int(np.ceil(n / elements_per_line(stride_bytes, line_bytes)))


@dataclass(frozen=True)
class View:
    """버퍼 위를 읽는 규칙. 버퍼 자체는 갖지 않는다."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int = 0

    def transpose(self) -> "View":
        """축을 뒤집는다. 버퍼를 한 바이트도 옮기지 않는다."""
        return View(self.shape[::-1], self.strides[::-1], self.offset)

    def step(self, k: int) -> "View":
        """첫 축을 k 개마다 하나씩 본다. 역시 규칙만 바뀐다."""
        n = (self.shape[0] + k - 1) // k
        return View((n,) + self.shape[1:], (self.strides[0] * k,) + self.strides[1:],
                    self.offset)

    def address(self, index: Sequence[int]) -> int:
        return self.offset + offset(index, self.strides)


def numpy_view(a: np.ndarray) -> View:
    """NumPy 배열이 들고 있는 규칙을 그대로 읽어 온다."""
    return View(tuple(a.shape), tuple(a.strides), 0)
