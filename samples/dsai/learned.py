"""8장. 학습된 인덱스. 누적 분포를 모델로 바꾸다.

정렬된 배열에서 키의 위치는 누적 분포 함수 값에 n 을 곱한 것이다.
그 함수를 조각 선형으로 근사하면
위치 예측이 나오고, 예측 오차의 최대값 안에서 이진 탐색으로 마무리한다.
조각 하나가 B-트리의 노드 하나를 대신한다.
"""
from __future__ import annotations

import numpy as np


class PiecewiseLinearIndex:
    """정의 8.9. 오차 상한 eps 의 조각 선형 모델. 조각은 (시작 키, 기울기,
    절편) 이다.
    """

    def __init__(self, keys: np.ndarray, eps: int):
        assert np.all(keys[:-1] <= keys[1:]), "정렬된 키"
        self.keys = keys
        self.eps = eps
        starts, slopes, icepts = [], [], []
        n = len(keys)
        i = 0
        while i < n:
            j = self._longest_segment(i)
            x0, x1 = keys[i], keys[j - 1]
            slope = (j - 1 - i) / (x1 - x0) if x1 > x0 else 0.0
            starts.append(x0)
            slopes.append(slope)
            icepts.append(i - slope * x0)
            i = j
        self.starts = np.array(starts, dtype=np.int64)
        self.slopes = np.array(slopes)
        self.icepts = np.array(icepts)

    def _longest_segment(self, i: int) -> int:
        """알고리즘 8.3. i 에서 시작해 양 끝을 잇는 직선이 모든 점을 eps 안에 두는
        가장 긴 구간의 끝."""
        n = len(self.keys)
        j = min(i + 2, n)
        while j < n:
            x0, x1 = self.keys[i], self.keys[j]
            if x1 == x0:
                j += 1
                continue
            slope = (j - i) / (x1 - x0)
            xs = self.keys[i:j + 1].astype(np.float64)
            pred = i + slope * (xs - x0)
            if np.max(np.abs(pred - np.arange(i, j + 1))) > self.eps:
                break
            j += 1
        return j

    def predict(self, k: np.ndarray) -> np.ndarray:
        seg = np.searchsorted(self.starts, k, side="right") - 1
        seg = np.clip(seg, 0, len(self.starts) - 1)
        pos = self.slopes[seg] * k + self.icepts[seg]
        return np.clip(np.rint(pos).astype(np.int64), 0, len(self.keys) - 1)

    def search(self, k: int) -> tuple[int, int]:
        """예측 위치 근처 2 eps 안에서 이진 탐색. 위치와 들여다본 원소 수."""
        p = int(self.predict(np.array([k]))[0])
        lo, hi = max(p - self.eps, 0), min(p + self.eps + 1, len(self.keys))
        j = int(np.searchsorted(self.keys[lo:hi], k, side="left"))
        return lo + j, int(np.ceil(np.log2(hi - lo + 1))) + 1

    def max_error(self) -> int:
        """전체 키에서의 최대 예측 오차. 정의의 eps 이하여야 한다."""
        return int(np.max(np.abs(self.predict(self.keys) - np.arange(len(self.keys)))))

    def nbytes(self) -> int:
        return self.starts.nbytes + self.slopes.nbytes + self.icepts.nbytes

    def nsegments(self) -> int:
        return len(self.starts)
