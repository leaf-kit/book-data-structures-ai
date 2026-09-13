"""13장. 인덱스 고르기. 재현율 곡선에서 목표를 넘는 첫 설정.

인덱스마다 설정 (nprobe, ef, 부분 수) 이 오름차순으로 있다.
설정이 커지면 재현율이 오르고 지연도 오른다.
목표 재현율을 정해 두고 그것을 넘는 첫 설정을 고르면 가장 싼 설정이다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def recall_at_k(fn: Callable, param, qs: np.ndarray, truth: list[set], k: int) -> float:
    """질의마다 fn(q, param) 의 앞 k 개가 정확한 답 k 개와 겹치는 비율. 그 평균."""
    hits = [len(set(np.asarray(fn(q, param)).tolist()[:k]) & t) / k
            for q, t in zip(qs, truth)]
    return float(np.mean(hits))


def recall_curve(fn: Callable, params: list, qs: np.ndarray, truth: list[set],
                 k: int, target: float):
    """알고리즘 13.4. 설정을 순서대로 시도해 재현율이 목표를 넘는 첫 설정과 그 재현율.
    못 넘으면 (None, 마지막 재현율).
    """
    rec = 0.0
    for p in params:
        rec = recall_at_k(fn, p, qs, truth, k)
        if rec >= target:
            return p, rec
    return None, rec
