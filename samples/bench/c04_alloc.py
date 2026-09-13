"""실험 4.2. 첫 맞춤 연속 할당과 자유 리스트의 할당 시간.

같은 요청 열을 넣는다. 첫 맞춤은 빈 구간 목록을 훑고, 자유 리스트는 머리를 뗀다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.alloc import ContiguousArena  # noqa: E402
from dsai.freelist import SlabPool  # noqa: E402

REPEAT = 5
STEPS = 20_000
LIVE = 256
SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]


def workload(rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.lognormal(5.0, 0.8, STEPS).astype(int), 16, 2048)


def run_firstfit(lengths: np.ndarray) -> None:
    arena = ContiguousArena(1 << 20)
    live: list[int] = []
    for k, ln in enumerate(lengths):
        if len(live) >= LIVE:
            arena.release(live.pop(0))
        if arena.alloc(k, int(ln)):
            live.append(k)


def run_slab(lengths: np.ndarray) -> None:
    pool = SlabPool(SIZES, LIVE + 8)
    live: list[tuple[int, int]] = []
    for ln in lengths:
        if len(live) >= LIVE:
            pool.free(*live.pop(0))
        live.append(pool.alloc(int(ln)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    lengths = workload(rng)
    rows = []
    t = median_ms(lambda: run_firstfit(lengths), REPEAT)
    rows.append(["첫 맞춤 연속", t, t * 1e6 / STEPS, 0.0])
    t = median_ms(lambda: run_slab(lengths), REPEAT)
    waste = SlabPool(SIZES, 1).internal_waste(lengths)
    rows.append(["자유 리스트 (등급 8개)", t, t * 1e6 / STEPS, waste * 100])
    cols = ["할당자", "시간 ms", "요청당 ns", "내부 단편화 퍼센트"]
    cond = (f"요청 {STEPS:,}개, 동시 {LIVE}개, 길이 16 에서 2048 로그정규. "
            "등급은 2의 거듭제곱")
    write_tsv("c04-alloc", cols, rows, "bench/c04_alloc.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
