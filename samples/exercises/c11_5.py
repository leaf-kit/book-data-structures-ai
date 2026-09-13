"""문제 11.5. CSR 에 간선 더하기. 다시 짜기와 따로 모아 두기."""
import numpy as np

from dsai.graph import CSRGraph


class RebuildCSR(CSRGraph):
    def __init__(self, n, src, dst):
        super().__init__(n, src, dst)
        self.src_all, self.dst_all = src.copy(), dst.copy()

    def add_edges(self, src, dst):
        self.src_all = np.concatenate([self.src_all, src])
        self.dst_all = np.concatenate([self.dst_all, dst])
        super().__init__(self.n, self.src_all, self.dst_all)      # 전부 다시 짠다


class BufferedCSR(CSRGraph):
    def __init__(self, n, src, dst, flush_at=1000):
        super().__init__(n, src, dst)
        self.src_all, self.dst_all = src.copy(), dst.copy()
        self.buf_src, self.buf_dst = [], []
        self.flush_at = flush_at

    def add_edges(self, src, dst):
        self.buf_src.extend(src.tolist())
        self.buf_dst.extend(dst.tolist())
        if len(self.buf_src) >= self.flush_at:
            self.flush()

    def flush(self):
        if self.buf_src:
            self.src_all = np.concatenate([self.src_all, np.array(self.buf_src)])
            self.dst_all = np.concatenate([self.dst_all, np.array(self.buf_dst)])
            self.buf_src, self.buf_dst = [], []
            CSRGraph.__init__(self, self.n, self.src_all, self.dst_all)

    def neighbors(self, u):
        base = super().neighbors(u)
        extra = [v for s, v in zip(self.buf_src, self.buf_dst) if s == u]
        if not extra:
            return base
        return np.concatenate([base, np.array(extra, dtype=np.int64)])


def test_both_agree_after_batches():
    rng = np.random.default_rng(0)
    n = 500
    s0, d0 = rng.integers(0, n, 2000), rng.integers(0, n, 2000)
    a, b = RebuildCSR(n, s0, d0), BufferedCSR(n, s0, d0, flush_at=300)
    for _ in range(10):
        s, d = rng.integers(0, n, 100), rng.integers(0, n, 100)
        a.add_edges(s, d)
        b.add_edges(s, d)
    for u in range(0, n, 37):
        assert sorted(a.neighbors(u).tolist()) == sorted(b.neighbors(u).tolist())
