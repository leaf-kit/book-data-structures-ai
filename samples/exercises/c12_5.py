"""문제 12.5. kd-트리의 잎 크기와 들여다본 점의 수.
잎이 크면 노드가 줄고 잎 안의 비교가 는다.
"""
import numpy as np

from dsai.kdtree import KDTree, brute_nearest


def seen_by_leaf_size(pts, qs, leaf_sizes):
    out = {}
    for b in leaf_sizes:
        tree = KDTree(pts, leaf_size=b)
        seen = []
        for q in qs:
            i, s = tree.nearest(q)
            assert i == brute_nearest(pts, q)
            seen.append(s)
        out[b] = float(np.mean(seen))
    return out


def test_leaf_size_tradeoff_in_low_dim():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((20_000, 4)).astype(np.float32)
    qs = rng.standard_normal((30, 4)).astype(np.float32)
    seen = seen_by_leaf_size(pts, qs, [4, 16, 64, 256])
    assert seen[4] < seen[256]                     # 잎이 크면 잎 안에서 더 본다
    assert all(v < 20_000 * 0.2 for v in seen.values())
