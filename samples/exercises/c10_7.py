"""문제 10.7. top-p 의 시작 k 를 이전 단계의 남은 수로. 두 배씩 늘리는 횟수가 준다."""
import numpy as np

from dsai.sampling import top_p_mask, top_p_mask_by_select


class AdaptiveTopP:
    def __init__(self, k0: int = 64):
        self.k = k0

    def mask(self, p: np.ndarray, top_p: float) -> np.ndarray:
        m = top_p_mask_by_select(p, top_p, self.k)
        self.k = max(8, int(m.sum() * 1.25))      # 다음은 이번 남은 수의 1.25 배부터
        return m


def test_adaptive_matches_sort_and_tracks_size():
    rng = np.random.default_rng(2)
    a = AdaptiveTopP()
    for _ in range(5):
        p = rng.dirichlet(np.ones(20_000) * 0.05)
        m = a.mask(p, 0.9)
        assert np.array_equal(m, top_p_mask(p, 0.9))
        assert a.k >= m.sum()
