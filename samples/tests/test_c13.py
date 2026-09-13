import numpy as np

from dsai.ivf import IVF, assign, kmeans
from dsai.ivfpq import IVFPQ
from dsai.nsw import clustered_points
from dsai.pq import PQ, quantization_error


def data():
    rng = np.random.default_rng(0)
    allp = clustered_points(6000, 16, 30, rng)
    return rng, allp[:5000], allp[5000:5050]


def test_kmeans_assign_and_ivf_recall():
    rng, pts, qs = data()
    cent = kmeans(pts, 64, rng, iters=8)
    a = assign(pts, cent)
    assert a.shape == (5000,) and a.max() < 64
    ivf = IVF(pts, cent)
    assert ivf.ptr[-1] == 5000
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:10].tolist()) for q in qs]
    rec = np.mean([len(set(ivf.search(q, 64, 10)[0].tolist()) & t) / 10
                   for q, t in zip(qs, truth)])
    assert rec > 0.99                                 # 전부 보면 정확하다
    rec1 = np.mean([len(set(ivf.search(q, 1, 10)[0].tolist()) & t) / 10
                    for q, t in zip(qs, truth)])
    assert rec1 < rec


def test_pq_error_drops_with_more_bytes():
    rng, pts, qs = data()
    e4 = quantization_error(PQ(pts, 4, rng, iters=5), pts)
    e8 = quantization_error(PQ(pts, 8, rng, iters=5), pts)
    assert 0 < e8 < e4 < 1
    pq = PQ(pts, 8, rng, iters=5)
    codes = pq.encode(pts)
    assert codes.dtype == np.uint8 and codes.shape == (5000, 8)
    d_adc = pq.adc(pq.table(qs[0]), codes)
    d_dec = ((pq.decode(codes) - qs[0]) ** 2).sum(axis=1)
    assert np.allclose(d_adc, d_dec, rtol=1e-4)      # 비대칭 거리는 복원 뒤 거리와 같다


def test_ivfpq_rerank_improves():
    rng, pts, qs = data()
    cent = kmeans(pts, 64, rng, iters=8)
    idx = IVFPQ(pts, cent, 4, rng)
    truth = [set(np.argsort(((pts - q) ** 2).sum(axis=1))[:10].tolist()) for q in qs]
    r0 = np.mean([len(set(idx.search(q, 16, 10, 0)[0].tolist()) & t) / 10
                  for q, t in zip(qs, truth)])
    r1 = np.mean([len(set(idx.search(q, 16, 10, 50)[0].tolist()) & t) / 10
                  for q, t in zip(qs, truth)])
    assert r1 >= r0
    assert idx.nbytes() < pts.nbytes
