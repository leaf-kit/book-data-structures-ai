"""문제 5.6. 테이프에 행렬 벡터 곱을 원시 연산으로 더한다."""
import numpy as np

from dsai.tape import Entry, Tape


class MatTape(Tape):
    def matvec(self, w: int, x: int) -> int:
        """y = W x. dy/dx 는 W 자체이고, 뒤로 흘릴 때는 W^T g 가 된다."""
        vw, vx = self.vals[w], self.vals[x]
        out = self.var(vw @ vx)
        self.entries.append(Entry(out, (x,), (vw,)))
        return out

    def backward(self, out: int) -> list[np.ndarray]:
        adj = [np.zeros_like(v) for v in self.vals]
        adj[out] = np.ones_like(self.vals[out])
        for e in reversed(self.entries):
            g = adj[e.out]
            for i, d in zip(e.inputs, e.local):
                if d.ndim == 2:            # 행렬 곱 항목. 전치를 곱한다
                    adj[i] = adj[i] + d.T @ g
                else:
                    adj[i] = adj[i] + g * d
        return adj


def test_matvec_gradient():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((4, 3))
    x = rng.standard_normal(3)
    t = MatTape()
    xi, wi = t.var(x), t.var(w)
    out = t.sum(t.tanh(t.matvec(wi, xi)))
    g = t.backward(out)[xi]
    y = w @ x
    expected = w.T @ (1 - np.tanh(y) ** 2)
    assert np.allclose(g, expected)
