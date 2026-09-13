"""실험 5.4. 정적 배칭과 연속 배칭. 같은 요청 열에서 처리량과 지연.

요청은 포아송으로 도착하고 길이는 로그정규다.
걸음 하나의 시간은 고정 값 더하기 배치 크기 곱하기 토큰 값이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.batching import (Request, simulate_continuous,  # noqa: E402
                           simulate_static, summarize)

N = 1000
FIXED_MS = 20.0        # 가중치를 한 번 읽는 값
PER_TOKEN_MS = 0.5     # 토큰 하나의 값
BATCHES = [8, 32, 128]
RATE = 0.004           # 밀리초당 요청 수. 0.004 면 초당 4 개


def make_requests(rng: np.random.Generator) -> list[Request]:
    gaps = rng.exponential(1.0 / RATE, N)
    arrivals = np.cumsum(gaps)
    tokens = np.clip(rng.lognormal(4.0, 0.7, N).astype(int), 4, 512)
    return [Request(float(a), int(t)) for a, t in zip(arrivals, tokens)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for b in BATCHES:
        for name, sim in [("정적", simulate_static), ("연속", simulate_continuous)]:
            reqs = make_requests(np.random.default_rng(SEED))
            sim(reqs, b, FIXED_MS, PER_TOKEN_MS)
            thr, mean, p95 = summarize(reqs)
            rows.append([b, name, thr, mean, p95])
    cols = ["배치 상한", "방식", "토큰/초", "평균 지연 ms", "p95 지연 ms"]
    cond = (f"요청 {N}개, 도착 밀리초당 {RATE}개, 길이 로그정규 4 에서 512 토큰. "
            f"걸음 시간 {FIXED_MS} + {PER_TOKEN_MS} x 배치 ms")
    write_tsv("c05-batch", cols, rows, "bench/c05_batch.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
