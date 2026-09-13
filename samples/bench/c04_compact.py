"""실험 4.3. 배열 안의 연결 리스트.
끼워 넣기로 흐트러진 순서를 압축하면 순회가 얼마나 빨라지는가.

끝에 붙이기만 하면 순회 순서와 메모리 순서가 같다. 중간에 끼우면 어긋난다.
압축은 둘을 다시 맞춘다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.linked import IndexList  # noqa: E402

REPEAT = 5
N = 4_000_000


def build(rng: np.random.Generator, inserts: int) -> IndexList:
    lst = IndexList(N + inserts + 1)
    for i in range(N):
        lst.append(float(i))
    slots = rng.integers(0, N, inserts)
    for s in slots:
        lst.insert_after(int(s), 1.0)
    return lst


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for inserts in [0, N // 10, N]:
        lst = build(rng, inserts)
        order = lst.order()
        t_before = median_ms(lambda: lst.gather_sum(order), REPEAT)
        lst.compact()
        order2 = lst.order()
        t_after = median_ms(lambda: lst.gather_sum(order2), REPEAT)
        arr = lst.val[:lst.size]
        t_array = median_ms(lambda: float(arr.sum()), REPEAT)
        rows.append([inserts, t_before, t_after, t_array])
    cols = ["끼운 수", "압축 전 ms", "압축 후 ms", "배열 sum ms"]
    cond = (f"노드 {N:,}개에 무작위 자리에 끼운 뒤 순회 순서로 모아 더함. "
            "압축은 순회 순서로 다시 놓기")
    write_tsv("c04-compact", cols, rows, "bench/c04_compact.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
