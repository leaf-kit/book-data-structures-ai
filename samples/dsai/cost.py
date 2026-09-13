"""0장. 균일 비용 모델과 그 바깥.

같은 n 번의 원소 접근이 접근 순서에 따라 얼마나 다른 값을 치르는지 재는 부품이다.
균일 비용 모델은 둘을 같은 값으로 세고, 이동량 모델은 다르게 센다.
"""
from __future__ import annotations

import numpy as np


def sequential_index(n: int) -> np.ndarray:
    """0, 1, 2, ... 순서. 연속 메모리를 앞에서 뒤로 훑는다."""
    return np.arange(n, dtype=np.int64)


def random_index(n: int, seed: int) -> np.ndarray:
    """같은 n 개 원소를 한 번씩, 다만 무작위 순서로 찾아간다."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n).astype(np.int64)


def gather_sum(values: np.ndarray, index: np.ndarray) -> float:
    """index 가 가리키는 순서대로 values 를 읽어 더한다.

    두 순서 모두 원소 n 개를 정확히 한 번씩 읽는다. 연산 수 W 는 같다.
    다른 것은 메모리 계층을 가로지르는 이동량 M 뿐이다.
    """
    return float(values[index].sum())


def blocks_touched(index: np.ndarray, elem_bytes: int, line_bytes: int) -> int:
    """이 접근 순서가 건드리는 캐시 라인 수를 센다.

    캐시가 라인 하나만 기억한다고 치는 가장 비관적인 셈이다.
    순차 접근은 n * elem_bytes / line_bytes 개, 무작위 접근은 n 개에 가깝다.
    """
    per_line = max(1, line_bytes // elem_bytes)
    lines = index // per_line
    changes = np.count_nonzero(np.diff(lines)) + 1
    return int(changes)


def moves_sequential(n: int, elem_bytes: int, line_bytes: int) -> int:
    """정리 0.1 의 상한. 순차 스캔이 옮기는 라인 수."""
    return -(-n * elem_bytes // line_bytes)


def moves_random_expected(n: int, elem_bytes: int, line_bytes: int,
                          cache_lines: int) -> float:
    """무작위 순서로 n 개를 읽을 때 기대되는 라인 이동 수.

    라인이 캐시에 남아 있을 확률을 cache_lines / total_lines 로 잡은 근사다.
    """
    total_lines = moves_sequential(n, elem_bytes, line_bytes)
    if total_lines <= cache_lines:
        return float(total_lines)
    hit = cache_lines / total_lines
    return n * (1.0 - hit) + cache_lines
