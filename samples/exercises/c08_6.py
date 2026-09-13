"""문제 8.6. B-트리 블록 안 탐색을 이진 탐색 대신 순차 비교로. 답은 같아야 한다."""
import numpy as np

from dsai.btree import StaticBTree


class LinearScanBTree(StaticBTree):
    def search(self, k: int) -> tuple[int, int]:
        pos, blocks = 0, 0
        for lvl in self.levels:
            lo = pos * self.B
            hi = min(lo + self.B, len(lvl))
            blocks += 1
            j = 0
            while lo + j + 1 < hi and lvl[lo + j + 1] <= k:     # 순차 비교
                j += 1
            pos = lo + j
        return pos, blocks


def test_linear_scan_agrees_with_binary():
    rng = np.random.default_rng(4)
    keys = np.sort(rng.choice(1 << 30, 200_000, replace=False).astype(np.int64))
    q = rng.choice(keys, 500)
    for B in (8, 16):
        a, b = StaticBTree(keys, B), LinearScanBTree(keys, B)
        for k in q:
            assert a.search(int(k)) == b.search(int(k))
