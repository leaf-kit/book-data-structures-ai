"""문제 12.7. 이웃 고르기 휴리스틱을 끄면. 가까운 것만 남기는 그래프의 재현율."""
import numpy as np

from dsai.nsw import NSW, clustered_points, d2


class NearestOnlyNSW(NSW):
    def select(self, v: int, cand: list[int], m: int) -> list[int]:
        """휴리스틱 없이 가까운 순으로 m 개. 긴 간선이 먼저 잘려 나간다."""
        pv = self.points[v]
        return sorted(cand, key=lambda w: d2(self.points, w, pv))[:m]


def recall_of(g, qs, truth, ef=32, k=10):
    hits = [len(set(g.search(q, ef)[0][:k]) & t) / k for q, t in zip(qs, truth)]
    return float(np.mean(hits))


def test_heuristic_matters():
    rng = np.random.default_rng(5)
    allp = clustered_points(3050, 16, 20, rng)
    pts, qs = allp[:3000], allp[3000:]
    order = rng.permutation(3000)
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:10].tolist()) for q in qs]
    with_h = recall_of(NSW(pts, 8).build(order), qs, truth)
    without = recall_of(NearestOnlyNSW(pts, 8).build(order), qs, truth)
    assert with_h > 0.85
    assert without < with_h            # 가까운 것만 남기면 재현율이 떨어진다
