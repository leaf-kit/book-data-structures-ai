"""문제 7.7. 보수적 갱신. 최소인 줄만 올린다.
답은 여전히 실제 이상이고 과대 추정이 준다.
"""
import numpy as np

from dsai.sketch import CountMinSketch, cms_depth_for, cms_width_for


class ConservativeCMS(CountMinSketch):
    def add(self, keys: np.ndarray) -> None:
        uniq, cnt = np.unique(keys, return_counts=True)
        s = self.slots(uniq)                                   # (u, d)
        rows = np.arange(self.d)
        cur = self.table[rows[None, :], s]                     # (u, d)
        target = cur.min(axis=1) + cnt              # 최소 줄이 개수만큼 오른 값
        new = np.maximum(cur, target[:, None])
        for r in range(self.d):
            np.maximum.at(self.table[r], s[:, r], new[:, r])


def test_conservative_never_under_and_less_over():
    rng = np.random.default_rng(3)
    ids = rng.zipf(1.3, 300_000) % 20_000
    keys = (ids * 7919 + 13).astype(np.int64)
    exact = np.bincount(ids, minlength=20_000)
    q = (np.arange(20_000) * 7919 + 13).astype(np.int64)
    w, d = cms_width_for(1e-3), cms_depth_for(1e-2)
    plain, cons = CountMinSketch(w, d), ConservativeCMS(w, d)
    plain.add(keys)
    cons.add(keys)
    ep, ec = plain.estimate(q), cons.estimate(q)
    assert np.all(ec >= exact)
    assert (ec - exact).sum() < (ep - exact).sum()          # 과대 추정 총량이 준다
