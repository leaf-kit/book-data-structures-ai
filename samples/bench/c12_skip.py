"""실험 12.2. 스킵 리스트의 지나온 노드 수와 메모리. p 를 바꿔 가며.

키 십만 개를 무작위 순서로 넣고 탐색이 지나는 노드 수를 잰다. 8장의 트립과 견준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.bst import Treap  # noqa: E402
from dsai.skiplist import SkipList  # noqa: E402

N = 100_000
Q = 2_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = rng.choice(1 << 40, N, replace=False).astype(np.int64)
    qs = rng.choice(keys, Q)
    rows = []
    for p in [0.25, 0.5]:
        sl = SkipList(N, p=p, seed=1)
        for k in keys:
            sl.insert(int(k))
        steps = [sl.search(int(q))[1] for q in qs]
        assert all(sl.search(int(q))[0] for q in qs[:100])
        bound = np.log(N) / np.log(1 / p) / p
        rows.append([f"스킵 리스트 p={p}", sl.top + 1, float(np.mean(steps)),
                     float(np.max(steps)), sl.nbytes() / 1e6, bound])
    t = Treap(N, seed=1)
    for k in keys:
        t.insert(int(k))
    probes = [t.search(int(q))[1] for q in qs]
    mb = (t.key.nbytes + t.left.nbytes + t.right.nbytes + t.prio.nbytes) / 1e6
    rows.append(["트립 (8장)", t.height(), float(np.mean(probes)),
                 float(np.max(probes)), mb, 2 * np.log(N)])
    cols = ["구조", "층 또는 높이", "평균 지나온 노드", "최대", "MB", "기대 상한"]
    cond = (f"무작위 40비트 키 {N:,}개, 탐색 {Q:,}개. "
            "기대 상한은 스킵 리스트 log n 나누기 p log(1/p)")
    write_tsv("c12-skip", cols, rows, "bench/c12_skip.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
