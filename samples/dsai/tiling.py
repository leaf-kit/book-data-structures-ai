"""1장. 접근 순서를 라인에 맞추는 법. 간격 훑기와 블록 전치.

같은 원소를 같은 횟수 읽어도 순서가 라인과 어긋나면 이동량이 열여섯 배가 된다.
블록으로 자르면 라인 하나를 가져와서 그 안의 원소를 다 쓴다.
"""
from __future__ import annotations

import numpy as np


def strided_sum(a: np.ndarray, k: int) -> float:
    """k 개마다 하나씩 읽어 더한다. k 가 라인 폭을 넘으면 원소마다 라인 하나다."""
    return float(a[::k].sum())


def transpose_copy(a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """전치한 뷰를 연속 메모리로 옮긴다. 읽기 순서가 열 방향이라 라인과 어긋난다."""
    if out is None:
        out = np.empty((a.shape[1], a.shape[0]), dtype=a.dtype)
    np.copyto(out, a.T)
    return out


def transpose_blocked(a: np.ndarray, block: int,
                      out: np.ndarray | None = None) -> np.ndarray:
    """block x block 타일 단위로 전치한다.

    타일 하나는 캐시에 들어가므로, 타일 안에서는 라인을 가져와서 원소를 다 쓴다.
    """
    n, m = a.shape
    if out is None:
        out = np.empty((m, n), dtype=a.dtype)
    for i in range(0, n, block):
        for j in range(0, m, block):
            out[j:j + block, i:i + block] = a[i:i + block, j:j + block].T
    return out


def tile_bytes(block: int, itemsize: int) -> int:
    """타일 하나가 차지하는 바이트. 입력 타일과 출력 타일 둘이 캐시에 들어야 한다."""
    return 2 * block * block * itemsize
