"""실험 10.2. 빔 폭과 찾은 점수, 그리고 한 단계의 값.

무작위 로그 확률 표 (어휘 64, 길이 8) 에서 전부 보기와 빔 1, 2, 4, 8 의
최선 점수를 견주고,
어휘 128,000, 빔 8 의 한 단계 시간을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.beam import beam_search, beam_step, exact_best  # noqa: E402

VOCAB, STEPS = 64, 8
TRIALS = 100


def make_model(rng, vocab, steps):
    table = rng.standard_normal((steps, vocab, vocab))     # 이전 토큰 -> 다음 토큰 점수
    table = table - np.log(np.exp(table).sum(axis=2, keepdims=True))

    def step_logp(prev, t):
        return table[t][prev]
    return step_logp, table


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    gaps = {b: [] for b in (1, 2, 4, 8)}
    found = {b: 0 for b in gaps}
    for _ in range(TRIALS):
        m, table = make_model(rng, VOCAB, STEPS)
        best = exact_best(table)
        for b in gaps:
            _, s = beam_search(m, b, STEPS, VOCAB)
            gaps[b].append(best - s)
            found[b] += int(abs(best - s) < 1e-9)
    for b in gaps:
        rows.append([b, VOCAB ** STEPS, b * VOCAB * STEPS, found[b] / TRIALS * 100,
                     float(np.mean(gaps[b]))])
    cols = ["빔 폭", "전부 보기 열 수", "빔이 본 열 수", "최선을 찾은 비율 퍼센트",
            "점수 차이 평균"]
    cond = (f"어휘 {VOCAB}, 길이 {STEPS}, 무작위 조건부 로그 확률 표 {TRIALS}개. "
            "전부 보기는 동적 계획법으로 같은 답을 낸다")
    write_tsv("c10-beam", cols, rows, "bench/c10_beam.py", cond, 1)

    V = 128_000
    scores = rng.standard_normal(8)
    logp = rng.standard_normal((8, V))
    ms_step = median_ms(lambda: beam_step(scores, logp, 8), 5)
    ms_sort = median_ms(lambda: np.sort(logp.ravel()), 5)
    rows2 = [["한 단계, 빔 8, 어휘 128,000", ms_step], ["같은 행렬 전부 정렬", ms_sort]]
    write_tsv("c10-beamstep", ["일", "ms"], rows2, "bench/c10_beam.py",
              "빔 8 x 어휘 128,000 의 점수 행렬에서 상위 8 개. 다섯 번 중앙값", 5)
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
