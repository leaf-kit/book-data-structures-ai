"""문제 11.7. 나눗셈과 exp 를 더한 스칼라 그래프. 팬아웃 셋의 합."""
import numpy as np

from dsai.topo import ScalarGraph


class RicherGraph(ScalarGraph):
    OPS = {"input": 0, "add": 1, "mul": 2, "tanh": 3, "div": 4, "exp": 5}

    def forward(self):
        for i in range(self.n):
            o, a, b = self.op[i], self.a[i], self.b[i]
            if o == 1:
                self.val[i] = self.val[a] + self.val[b]
            elif o == 2:
                self.val[i] = self.val[a] * self.val[b]
            elif o == 3:
                self.val[i] = np.tanh(self.val[a])
            elif o == 4:
                self.val[i] = self.val[a] / self.val[b]
            elif o == 5:
                self.val[i] = np.exp(self.val[a])

    def backward(self, out):
        grad = np.zeros(self.n)
        grad[out] = 1.0
        for i in range(out, -1, -1):
            o, g, a, b = self.op[i], grad[i], self.a[i], self.b[i]
            if g == 0.0:
                continue
            if o == 1:
                grad[a] += g
                grad[b] += g
            elif o == 2:
                grad[a] += g * self.val[b]
                grad[b] += g * self.val[a]
            elif o == 3:
                grad[a] += g * (1.0 - self.val[i] ** 2)
            elif o == 4:
                grad[a] += g / self.val[b]
                grad[b] += -g * self.val[a] / self.val[b] ** 2
            elif o == 5:
                grad[a] += g * self.val[i]
        return grad


def test_fanout_three_matches_finite_difference():
    g = RicherGraph(12)
    x = g.node("input", value=0.7)
    y = g.node("input", value=1.9)
    e = g.node("exp", x)
    d = g.node("div", x, y)
    t = g.node("tanh", x)                # x 가 세 곳에 쓰인다
    s1 = g.node("add", e, d)
    out = g.node("mul", s1, t)
    g.forward()
    grad = g.backward(out)
    eps = 1e-6
    for i in (x, y):
        g.val[i] += eps
        g.forward()
        up = g.val[out]
        g.val[i] -= 2 * eps
        g.forward()
        dn = g.val[out]
        g.val[i] += eps
        assert abs((up - dn) / (2 * eps) - grad[i]) < 1e-6
