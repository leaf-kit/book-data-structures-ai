"""실험 10.4. 도착 순서, 짧은 것 먼저, 나이를 더한 우선순위. 평균 대기와 최대 대기.

요청 5,000 개가 푸아송으로 오고 길이는 지프 분포다. 한 번에 하나를 처리한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.scheduler import simulate  # noqa: E402

N = 5000
RATE = 2000.0            # 토큰/초


def main() -> None:
    rng = np.random.default_rng(SEED)
    lengths = np.minimum(rng.zipf(1.6, N) * 32, 4096)
    load = 0.9
    gap = lengths.mean() / RATE / load
    arrivals_t = np.cumsum(rng.exponential(gap, N))
    arrivals = list(zip(arrivals_t.tolist(), lengths.tolist()))
    rows = []
    policies = [("도착 순서", "fifo", 0.0), ("짧은 것 먼저", "sjf", 0.0),
                ("짧은 것 먼저 + 나이", "aged", 200.0)]
    for name, policy, aging in policies:
        mean_w, max_w, p99 = simulate(arrivals, policy, RATE, aging)
        rows.append([name, mean_w * 1e3, p99 * 1e3, max_w * 1e3])
    cols = ["정책", "평균 대기 ms", "99 퍼센타일 ms", "최대 대기 ms"]
    cond = (f"요청 {N:,}개, 푸아송 도착, 부하 {load}, "
            f"길이 지프 (32 토큰 단위, 최대 4,096). 처리량 {RATE:.0f} 토큰/초. "
            "나이 계수는 초당 200 토큰")
    write_tsv("c10-scheduler", cols, rows, "bench/c10_scheduler.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
