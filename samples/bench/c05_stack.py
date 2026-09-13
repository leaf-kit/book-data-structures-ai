"""실험 5.1. 재귀와 명시적 스택. 깊이에 따른 시간과 상한.

같은 트리의 깊이를 재귀로 재고 명시적 스택으로 잰다. 재귀는 어느 깊이에서 멈춘다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import median_ms, write_tsv  # noqa: E402
from dsai.stack import chain_children, depth_iterative, depth_recursive  # noqa: E402

REPEAT = 5
DEPTHS = [100, 1_000, 10_000, 100_000, 1_000_000]


def main() -> None:
    rows = []
    for n in DEPTHS:
        children = chain_children(n)
        t_iter = median_ms(lambda: depth_iterative(children), REPEAT)
        try:
            t_rec = median_ms(lambda: depth_recursive(children), REPEAT)
            rec = f"{t_rec:.2f}"
        except RecursionError:
            rec = "실패"
        rows.append([n, rec, t_iter])
    cond = f"자식이 하나뿐인 트리. 파이썬 기본 재귀 상한 {sys.getrecursionlimit()}"
    write_tsv("c05-stack", ["깊이", "재귀 ms", "명시적 스택 ms"], rows,
              "bench/c05_stack.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
