"""실험 8.2. 차수와 높이, 블록 수. 정렬된 키 천만 개 위의 정적 B-트리.

차수를 2 (이진 탐색 트리 꼴) 에서 512 까지 바꾸며 높이와 탐색당 블록 수,
내부 노드 메모리를 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.btree import StaticBTree, expected_height  # noqa: E402

N = 10_000_000
Q = 20_000
FANOUTS = [2, 8, 16, 64, 512]


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = np.sort(rng.choice(1 << 40, N, replace=False).astype(np.int64))
    queries = rng.choice(keys, Q)
    rows = []
    for B in FANOUTS:
        t = StaticBTree(keys, B)
        blocks = [t.search(int(q))[1] for q in queries[:2000]]
        ms = median_ms(lambda: [t.search(int(q)) for q in queries[:2000]], 3)
        rows.append([B, t.height(), expected_height(N, B), float(np.mean(blocks)),
                     t.nbytes() / 1e6, ms / 2000 * 1e3])
    ms = median_ms(lambda: [int(np.searchsorted(keys, q)) for q in queries[:2000]], 3)
    rows.append([0, int(np.ceil(np.log2(N))), int(np.ceil(np.log2(N))),
                 float(np.ceil(np.log2(N))), 0.0, ms / 2000 * 1e3])
    cols = ["차수", "높이", "예측 높이", "탐색당 블록", "내부 노드 MB", "탐색당 us"]
    cond = (f"정렬된 40비트 키 {N:,}개, 탐색 2,000개의 평균. "
            "차수 0 은 배열 전체의 이진 탐색이고 블록은 라인 수가 아니라 단계 수다")
    write_tsv("c08-btree", cols, rows, "bench/c08_btree.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
