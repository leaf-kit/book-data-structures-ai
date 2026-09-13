"""16장. 열 지향 저장과 메모리 맵.

행마다 레코드를 두면 열 하나를 읽는 데 전부를 읽는다. 열마다 배열을 두면 그 열만 읽는다.
메모리 맵은 파일을 열지 않고 배열로 본다. 페이지가 처음 닿을 때 읽히므로,
읽은 만큼만 이동이다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_row_major(path: Path, table: dict[str, np.ndarray]) -> np.ndarray:
    """디딤돌. 레코드 하나가 모든 열을 잇달아 갖는 구조체 배열."""
    names = list(table)
    dtype = np.dtype([(n, table[n].dtype) for n in names])
    rows = np.empty(len(table[names[0]]), dtype=dtype)
    for n in names:
        rows[n] = table[n]
    rows.tofile(path)
    return rows


def write_column_major(dirpath: Path, table: dict[str, np.ndarray]) -> None:
    """정의 16.4. 열마다 파일 하나. 같은 열이 연속이다."""
    dirpath.mkdir(exist_ok=True)
    for n, col in table.items():
        col.tofile(dirpath / f"{n}.bin")


def column_sum_row_major(path: Path, dtype: np.dtype, name: str) -> float:
    """행 파일에서 열 하나를 읽는다. 메모리 맵이어도 레코드마다 라인 하나다."""
    rows = np.memmap(path, dtype=dtype, mode="r")
    return float(rows[name].astype(np.float64).sum())


def column_sum_column_major(dirpath: Path, name: str, dtype: np.dtype) -> float:
    """열 파일 하나만 읽는다. 순차 접근이고 그 열의 바이트만 이동이다."""
    col = np.memmap(dirpath / f"{name}.bin", dtype=dtype, mode="r")
    return float(col.astype(np.float64).sum())


def bytes_touched(n_rows: int, row_bytes: int, col_bytes: int, layout: str) -> int:
    """명제 16.4. 열 하나를 읽는 데 닿는 바이트. 행 저장은 전부, 열 저장은 그 열만."""
    return n_rows * (row_bytes if layout == "row" else col_bytes)
