"""실험 8.1. 넣는 순서와 높이. 무작위 순서, 정렬 순서, 그리고 트립.

키 십만 개를 같은 트리에 세 방식으로 넣고 높이와 탐색 노드 수를 잰다.
정렬 순서는 만 개만 넣는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.bst import BST, Treap  # noqa: E402

N = 100_000
N_SORTED = 10_000
Q = 2_000


def measure(tree, keys, queries):
    for k in keys:
        tree.insert(int(k))
    probes = [tree.search(int(q))[1] for q in queries]
    return tree.height(), float(np.mean(probes))


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = rng.choice(1 << 40, N, replace=False).astype(np.int64)
    queries = rng.choice(keys, Q)
    rows = []
    h, p = measure(BST(N), keys, queries)
    rows.append(["무작위 순서 BST", N, h, p, 2 * np.log2(N)])
    sorted_keys = np.sort(keys[:N_SORTED])
    h, p = measure(BST(N_SORTED), sorted_keys, rng.choice(sorted_keys, Q))
    rows.append(["정렬 순서 BST", N_SORTED, h, p, 2 * np.log2(N_SORTED)])
    h, p = measure(Treap(N_SORTED, seed=1), sorted_keys, rng.choice(sorted_keys, Q))
    rows.append(["정렬 순서 트립", N_SORTED, h, p, 2 * np.log2(N_SORTED)])
    h, p = measure(Treap(N, seed=1), keys, queries)
    rows.append(["무작위 순서 트립", N, h, p, 2 * np.log2(N)])
    rows.append(["완전 균형 (하한)", N, int(np.ceil(np.log2(N + 1))), np.log2(N) - 1,
                 np.log2(N)])
    cols = ["넣는 방식", "키 수", "높이", "평균 탐색 노드", "2 log2 n"]
    cond = (f"무작위 40비트 정수 키. 탐색은 들어 있는 키 {Q:,}개. "
            f"정렬 순서는 {N_SORTED:,}개만")
    write_tsv("c08-height", cols, rows, "bench/c08_height.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
