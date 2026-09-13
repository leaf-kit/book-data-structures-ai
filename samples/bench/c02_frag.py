"""실험 2.2. 길이가 제각각인 요청이 들어오고 나갈 때, 연속 할당과 페이지 할당의 단편화.

같은 요청 열을 두 할당자에 넣는다. 연속 할당은 총합이 남아도 자리가 없어 실패하고,
페이지 할당은 실패 대신 마지막 페이지의 자투리를 남긴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.alloc import ContiguousArena, PagedArena  # noqa: E402

TOTAL = 1 << 17      # 토큰 자리 13만 개
STEPS = 20_000
LIVES = [48, 64]     # 동시에 살아 있는 요청 수. 평균 부하 60 과 80 퍼센트
PAGES = [16, 256]


def request_lengths(rng: np.random.Generator, n: int) -> np.ndarray:
    """긴 꼬리 분포. 대부분 짧고 가끔 아주 길다."""
    return np.clip(rng.lognormal(7.0, 0.9, n).astype(int), 16, TOTAL // 8)


def run(arena, lengths: np.ndarray, live_max: int) -> tuple[int, float]:
    live: list[int] = []
    fails = 0
    frag_sum = 0.0
    for key, ln in enumerate(lengths):
        if len(live) >= live_max:
            arena.release(live.pop(0))
        if not arena.alloc(key, int(ln)):
            fails += 1
        else:
            live.append(key)
        if isinstance(arena, ContiguousArena):
            frag_sum += arena.external_fragmentation()
        else:
            frag_sum += arena.internal_fragmentation()
    return fails, frag_sum / len(lengths)


def main() -> None:
    rng = np.random.default_rng(SEED)
    lengths = request_lengths(rng, STEPS)
    rows = []
    for live in LIVES:
        load = round(live * lengths.mean() / TOTAL * 100)
        fails, frag = run(ContiguousArena(TOTAL), lengths, live)
        rows.append([load, "연속 첫 맞춤", fails, frag * 100, 0.0])
        for p in PAGES:
            fails, frag = run(PagedArena(TOTAL, p), lengths, live)
            rows.append([load, f"페이지 {p}", fails, 0.0, frag * 100])
    cols = ["부하", "할당자", "실패", "외부 퍼센트", "내부 퍼센트"]
    cond = (f"자리 {TOTAL:,}개, 요청 {STEPS:,}개, 길이는 로그정규 분포. "
            "부하는 평균 점유율")
    write_tsv("c02-frag", cols, rows, "bench/c02_frag.py", cond, 1)
    print(f"평균 길이 {lengths.mean():.0f}, 최대 {lengths.max()}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
