"""실험 8.3. 학습된 인덱스의 조각 수와 메모리. 오차 상한과 키 분포에 따라.

균일 분포와 지프 (한쪽으로 쏠린) 분포의 키 백만 개에서 조각 선형 인덱스를 세우고
같은 높이의 B-트리 내부 노드 메모리와 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.btree import StaticBTree  # noqa: E402
from dsai.learned import PiecewiseLinearIndex  # noqa: E402

N = 1_000_000
EPS = [8, 32, 128]


def datasets(rng):
    uni = np.sort(rng.choice(1 << 40, N, replace=False).astype(np.int64))
    z = np.sort(np.unique(np.cumsum(rng.zipf(1.5, 2 * N)[:N]).astype(np.int64)))[:N]
    return {"균일": uni, "쏠림": z}


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, keys in datasets(rng).items():
        for eps in EPS:
            idx = PiecewiseLinearIndex(keys, eps)
            q = rng.choice(keys, 2000)
            probes = [idx.search(int(k))[1] for k in q]
            assert all(keys[idx.search(int(k))[0]] == k for k in q[:200])
            rows.append([name, eps, idx.nsegments(), idx.max_error(),
                         idx.nbytes() / 1e3, float(np.mean(probes))])
        bt = StaticBTree(keys, 64)
        rows.append([name, 0, len(bt.levels[-2]), 0, bt.nbytes() / 1e3,
                     float(bt.height())])
    cols = ["분포", "오차 상한", "조각 또는 노드 수", "실측 최대 오차", "KB",
            "탐색당 단계"]
    cond = (f"키 {N:,}개. 오차 상한 0 은 차수 64 의 B-트리 내부 노드이고, "
            "그때 조각 수 열은 잎 바로 위 노드의 키 수다")
    write_tsv("c08-learned", cols, rows, "bench/c08_learned.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
