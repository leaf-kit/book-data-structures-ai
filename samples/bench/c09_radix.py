"""실험 9.3. 접두사 공유 캐시.
같은 시스템 프롬프트를 나누는 대화에서 다시 계산하는 토큰 수.

사용자 200 명이 시스템 프롬프트 512 토큰을 공유하고,
각자 다섯 턴에 걸쳐 128 토큰씩 붙인다.
요청은 사용자를 섞어 온다. 캐시 용량을 바꾸며 다시 계산한 토큰과 노드 수를 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.radix import RadixCache  # noqa: E402

USERS = 200
TURNS = 5
SYSTEM = 512
TURN = 128


def requests(rng):
    system = rng.integers(0, 32000, SYSTEM).tolist()
    hist = [list(system) for _ in range(USERS)]
    order = []
    for t in range(TURNS):
        for u in rng.permutation(USERS):
            hist[u] = hist[u] + rng.integers(0, 32000, TURN).tolist()
            order.append(list(hist[u]))
    return order


def main() -> None:
    rng = np.random.default_rng(SEED)
    reqs = requests(rng)
    total = sum(len(r) for r in reqs)
    rows = [["캐시 없음", 0, total, 0.0, 0]]
    for cap in [20_000, 80_000, 400_000]:
        cache = RadixCache(cap)
        computed = 0
        for r in reqs:
            computed += cache.insert(r)
        saved = (1 - computed / total) * 100
        rows.append([f"용량 {cap // 1000}K", cap, computed, saved, cache.node_count()])
    cols = ["캐시", "용량 토큰", "계산한 토큰", "절약 퍼센트", "노드 수"]
    cond = (f"사용자 {USERS}명, 시스템 프롬프트 {SYSTEM} 토큰, "
            f"턴 {TURNS}개 x {TURN} 토큰. 요청 {len(reqs):,}개, 토큰 합 {total:,}")
    write_tsv("c09-radix", cols, rows, "bench/c09_radix.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
