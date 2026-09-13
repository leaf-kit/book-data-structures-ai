"""문제 13.6. 무작위 직교 회전을 먼저 하는 곱 양자화."""
import numpy as np

from dsai.nsw import clustered_points
from dsai.pq import PQ, quantization_error


class RotatedPQ:
    def __init__(self, x: np.ndarray, m: int, rng: np.random.Generator, iters: int = 5):
        d = x.shape[1]
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        self.rot = q.astype(x.dtype)
        self.pq = PQ(x @ self.rot, m, rng, iters)

    def error(self, x: np.ndarray) -> float:
        return quantization_error(self.pq, x @ self.rot)


def test_rotation_changes_error_little_for_isotropic_data():
    rng = np.random.default_rng(0)
    x = clustered_points(8000, 32, 50, rng)
    plain = quantization_error(PQ(x, 8, np.random.default_rng(1), iters=5), x)
    rot = RotatedPQ(x, 8, np.random.default_rng(1)).error(x)
    assert 0 < rot < 1 and abs(rot - plain) < 0.5 * plain    # 같은 자릿수
