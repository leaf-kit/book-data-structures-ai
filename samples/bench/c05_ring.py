"""실험 5.3. 창 평균. 매번 다시 더하기와 링 버퍼 누적합.

창 길이 w 를 바꿔 가며 값 n 개를 흘려 넣는다. 다시 더하기는 갱신마다 w 개를 읽고,
링 버퍼는 둘만 읽는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.ring import WindowMean, window_mean_recompute  # noqa: E402

REPEAT = 5
N = 20_000
WINDOWS = [16, 256, 4096]


def ring_all(xs: np.ndarray, w: int) -> np.ndarray:
    wm = WindowMean(w)
    out = np.empty(len(xs))
    for i, v in enumerate(xs):
        out[i] = wm.push(float(v))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    xs = rng.standard_normal(N)
    rows = []
    for w in WINDOWS:
        t_re = median_ms(lambda: window_mean_recompute(xs, w), REPEAT)
        t_ring = median_ms(lambda: ring_all(xs, w), REPEAT)
        rows.append([w, t_re, t_ring, t_re / t_ring])
    cols = ["창 길이", "다시 더하기 ms", "링 버퍼 ms", "비율"]
    write_tsv("c05-ring", cols, rows, "bench/c05_ring.py",
              f"값 {N:,}개를 흘려 넣으며 창 평균을 매번 갱신", REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
