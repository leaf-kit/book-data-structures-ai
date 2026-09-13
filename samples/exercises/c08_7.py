"""문제 8.7. 두 층 학습 인덱스. 조각 시작 키 위에 같은 인덱스를 한 번 더 세운다."""
import numpy as np

from dsai.learned import PiecewiseLinearIndex


class TwoLevelIndex:
    def __init__(self, keys: np.ndarray, eps: int):
        self.keys = keys
        self.eps = eps
        self.bottom = PiecewiseLinearIndex(keys, eps)
        self.top = PiecewiseLinearIndex(self.bottom.starts, eps)

    def search(self, k: int) -> tuple[int, int]:
        """위 층으로 조각 번호를 예측해 창 안에서 찾고,
        아래 층 조각으로 위치를 예측한다.
        """
        s, steps_top = self.top.search(k)                 # k 이상인 첫 조각 시작
        starts = self.bottom.starts
        seg = s if s < len(starts) and starts[s] == k else max(s - 1, 0)
        p = self.bottom.slopes[seg] * k + self.bottom.icepts[seg]
        p = int(np.clip(np.rint(p), 0, len(self.keys) - 1))
        lo, hi = max(p - self.eps, 0), min(p + self.eps + 1, len(self.keys))
        j = int(np.searchsorted(self.keys[lo:hi], k))
        return lo + j, steps_top + int(np.ceil(np.log2(hi - lo + 1)))


def test_two_level_exact_and_cheaper():
    rng = np.random.default_rng(0)
    keys = np.sort(rng.choice(1 << 34, 300_000, replace=False).astype(np.int64))
    idx = TwoLevelIndex(keys, eps=32)
    assert idx.top.nsegments() < idx.bottom.nsegments() // 4
    for k in keys[::1013]:
        pos, steps = idx.search(int(k))
        assert keys[pos] == k
        assert steps < np.log2(idx.bottom.nsegments()) + np.log2(65) + 2
