"""3장. 양자화된 텐서. 값 배열의 타입을 작게 바꾸고 그 대가로 오차를 받는다.

실수 하나를 정수 몇 비트와 스케일 하나로 적는다.
스케일을 텐서 전체에 하나 두면 큰 값 하나가 나머지 전부의 정밀도를 깎는다.
블록마다 스케일을 두면 그 피해가 블록 안에 갇힌다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QTensor:
    """정의 3.5. 정수 코드, 블록마다의 스케일, 비트 수, 블록 크기."""

    codes: np.ndarray      # int8 로 담되 bits 만큼만 쓴다
    scales: np.ndarray     # (블록 수,) float32
    bits: int
    block: int
    shape: tuple[int, ...]

    def nbytes(self) -> int:
        return self.codes.size * self.bits // 8 + self.scales.nbytes


def quantize(x: np.ndarray, bits: int, block: int) -> QTensor:
    """알고리즘 3.3. 블록마다 최대 절댓값으로 스케일을 정하고 반올림한다."""
    flat = x.reshape(-1).astype(np.float32)
    assert flat.size % block == 0
    groups = flat.reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(groups).max(axis=1)
    scales = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    codes = np.clip(np.rint(groups / scales[:, None]), -qmax - 1, qmax).astype(np.int8)
    return QTensor(codes, scales, bits, block, x.shape)


def dequantize(q: QTensor) -> np.ndarray:
    return (q.codes.astype(np.float32) * q.scales[:, None]).reshape(q.shape)


def quant_error(x: np.ndarray, q: QTensor) -> float:
    """상대 제곱 오차. 정리 3.2 의 균등 양자화 오차와 견준다."""
    d = dequantize(q) - x
    return float((d * d).sum() / (x * x).sum())


def step_size(q: QTensor) -> np.ndarray:
    """블록마다의 양자화 간격. 정리 3.2 의 델타다."""
    return q.scales


def matvec_dequant(q: QTensor, x: np.ndarray) -> np.ndarray:
    """정수 코드를 읽어 스케일을 곱하며 곱한다. 읽는 바이트는 bits/8 배로 준다."""
    w = dequantize(q)
    return w @ x
