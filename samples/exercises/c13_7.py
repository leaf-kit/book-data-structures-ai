"""문제 13.7. 군집별 거리표를 한꺼번에. (nprobe, m, 256) 표를 행렬 곱으로."""
import numpy as np

from dsai.ivf import kmeans
from dsai.ivfpq import IVFPQ
from dsai.nsw import clustered_points


def tables_batched(idx: IVFPQ, q: np.ndarray, lists: np.ndarray) -> np.ndarray:
    """잔차 질의 (nprobe, d) 와 코드북 (m, 256, ds) 로 표 (nprobe, m,
    256) 을 한 번에.
    """
    r = q[None, :] - idx.c[lists]                                  # (nprobe, d)
    m, ds = idx.pq.m, idx.pq.ds
    parts = r.reshape(len(lists), m, ds)                          # (nprobe, m, ds)
    books = idx.pq.books                                          # (m, 256, ds)
    rr = (parts ** 2).sum(axis=2)[:, :, None]
    bb = (books ** 2).sum(axis=2)[None, :, :]
    cross = np.einsum("pmd,mkd->pmk", parts, books)
    return rr - 2.0 * cross + bb


def test_batched_tables_match_loop():
    rng = np.random.default_rng(0)
    x = clustered_points(5000, 16, 30, rng)
    cent = kmeans(x, 32, rng, iters=5)
    idx = IVFPQ(x, cent, 4, rng)
    q = x[3] + 0.01
    lists = np.array([0, 5, 9])
    batched = tables_batched(idx, q, lists)
    for i, l in enumerate(lists):
        assert np.allclose(batched[i], idx.pq.table(q - idx.c[l]), atol=1e-4)
