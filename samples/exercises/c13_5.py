"""문제 13.5. k-평균++ 초기화. 첫 중심은 무작위, 다음은 거리 제곱에 비례해 뽑는다."""
import numpy as np

from dsai.ivf import assign, kmeans
from dsai.nsw import clustered_points


def kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    c = [x[rng.integers(len(x))]]
    d2 = ((x - c[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        p = d2 / d2.sum()
        i = int(rng.choice(len(x), p=p))              # 10장의 누적합 뽑기
        c.append(x[i])
        d2 = np.minimum(d2, ((x - x[i]) ** 2).sum(axis=1))
    return np.stack(c)


def lloyd_from(x, c, iters):
    c = c.copy()
    for _ in range(iters):
        a = assign(x, c)
        for j in range(len(c)):
            m = a == j
            if m.any():
                c[j] = x[m].mean(axis=0)
    return c


def objective(x, c):
    a = assign(x, c)
    return float(((x - c[a]) ** 2).sum())


def test_pp_init_is_not_worse():
    rng = np.random.default_rng(0)
    x = clustered_points(6000, 16, 40, rng)
    c_rand = kmeans(x, 40, np.random.default_rng(1), iters=5)
    c_pp = lloyd_from(x, kmeans_pp_init(x, 40, np.random.default_rng(1)), 5)
    assert objective(x, c_pp) <= objective(x, c_rand) * 1.05
