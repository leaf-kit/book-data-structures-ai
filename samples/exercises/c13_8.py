"""문제 13.8. 재현율 목표를 바꾸면 이기는 인덱스가 바뀌는가. 작은 데이터로 확인."""
import numpy as np

from dsai.ivf import IVF, kmeans
from dsai.nsw import NSW, clustered_points


def first_passing(fn, params, qs, truth, target, k=10):
    for p in params:
        rec = np.mean([len(set(np.asarray(fn(q, p)).tolist()[:k]) & t) / k
                       for q, t in zip(qs, truth)])
        if rec >= target:
            return p, rec
    return None, rec


def test_targets_change_settings():
    rng = np.random.default_rng(0)
    allp = clustered_points(4050, 16, 40, rng)
    pts, qs = allp[:4000], allp[4000:]
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:10].tolist()) for q in qs]
    ivf = IVF(pts, kmeans(pts, 64, rng, iters=5))
    g = NSW(pts, 8).build(rng.permutation(4000))
    probes, efs = [1, 2, 4, 8, 16, 32], [10, 20, 40, 80]
    ivf10 = lambda q, p: ivf.search(q, p, 10)[0]  # noqa: E731
    np_lo, _ = first_passing(ivf10, probes, qs, truth, 0.8)
    np_hi, _ = first_passing(ivf10, probes, qs, truth, 0.99)
    ef_lo, _ = first_passing(lambda q, ef: g.search(q, ef)[0], efs, qs, truth, 0.8)
    ef_hi, _ = first_passing(lambda q, ef: g.search(q, ef)[0], efs, qs, truth, 0.99)
    assert np_lo is not None and ef_lo is not None
    # 목표가 오르면 설정이 커진다
    assert (np_hi or 99) >= np_lo and (ef_hi or 999) >= ef_lo
