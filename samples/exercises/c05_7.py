"""문제 5.7. 창 분산을 상수 시간에. 합과 제곱합 둘을 든다."""
import numpy as np

from dsai.ring import RingBuffer


class WindowVar:
    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.s1 = 0.0
        self.s2 = 0.0

    def push(self, v: float) -> float:
        if self.ring.n == self.ring.cap:
            old = float(self.ring.buf[self.ring.head])
            self.s1 -= old
            self.s2 -= old * old
        self.ring.push(v)
        self.s1 += v
        self.s2 += v * v
        n = self.ring.n
        return self.s2 / n - (self.s1 / n) ** 2


def test_window_var_matches_numpy():
    xs = np.random.default_rng(7).standard_normal(300)
    wv = WindowVar(16)
    got = np.array([wv.push(float(v)) for v in xs])
    want = np.array([xs[max(0, i - 15):i + 1].var() for i in range(300)])
    assert np.allclose(got, want, atol=1e-9)
